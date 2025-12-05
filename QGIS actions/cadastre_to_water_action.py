import json
import math
from collections import defaultdict
import urllib.request
import urllib.parse

from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsFields, QgsField, QgsSymbol, QgsRuleBasedRenderer, Qgis, QgsRectangle,
    QgsSpatialIndex, QgsWkbTypes
)
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor
import processing


# ----------------------------- helpers ---------------------------------

def _segment_type(piiripunkt_from, piiripunkt_to):
    water_val = "VEEKOGU_TELJE_PUNKT"
    from_is_water = piiripunkt_from == water_val
    to_is_water = piiripunkt_to == water_val

    if from_is_water and to_is_water:
        return "water"
    if from_is_water != to_is_water:
        return "interpolate"
    return "land"


def qvariant_type(value):
    if isinstance(value, bool):
        return QVariant.Bool
    if isinstance(value, int):
        return QVariant.LongLong
    if isinstance(value, float):
        return QVariant.Double
    return QVariant.String


def polygon_boundary_lines(poly_geom):
    """
    Return all polygon boundary LineString geometries (outer + holes).
    """
    boundaries = []

    if poly_geom.isMultipart():
        polys = poly_geom.asMultiPolygon()
        for poly in polys:
            for ring in poly:
                boundaries.append(QgsGeometry.fromPolylineXY(ring))
    else:
        poly = poly_geom.asPolygon()
        for ring in poly:
            boundaries.append(QgsGeometry.fromPolylineXY(ring))

    return boundaries


def polyline_from_geom(geom):
    if geom.isMultipart():
        m = geom.asMultiPolyline()
        return m[0] if m else None
    else:
        return geom.asPolyline()


def get_points_from_geometry(geom):
    pts = []
    gtype = QgsWkbTypes.geometryType(geom.wkbType())

    if gtype == QgsWkbTypes.PointGeometry:
        if QgsWkbTypes.isMultiType(geom.wkbType()):
            for p in geom.asMultiPoint():
                pts.append(QgsPointXY(p))
        else:
            pts.append(QgsPointXY(geom.asPoint()))

    elif gtype == QgsWkbTypes.LineGeometry:
        if QgsWkbTypes.isMultiType(geom.wkbType()):
            for pl in geom.asMultiPolyline():
                for p in pl:
                    pts.append(QgsPointXY(p))
        else:
            for p in geom.asPolyline():
                pts.append(QgsPointXY(p))

    return pts


def point_key(pt, decimals):
    return (round(pt.x(), decimals), round(pt.y(), decimals))


def closest_segment_info(pt, target_index, target_feats):
    """
    Find closest segment on snap target and return dict or None:
    {
      'feat_id': feature id,
      'sub_idx': boundary index (for polygon), else 0,
      'after_vertex': after_vertex index,
      'snapped_pt': QgsPointXY,
      'dist': float
    }
    """
    cand_ids = target_index.nearestNeighbor(pt, 10)
    if not cand_ids:
        return None

    best = None

    for tid in cand_ids:
        tfeat = target_feats[tid]
        tgeom = tfeat.geometry()

        if QgsWkbTypes.geometryType(tgeom.wkbType()) == QgsWkbTypes.PolygonGeometry:
            subgeoms = polygon_boundary_lines(tgeom)
        else:
            subgeoms = [tgeom]

        for si, sg in enumerate(subgeoms):
            ctx = sg.closestSegmentWithContext(pt)
            if ctx is None:
                continue
            dist, snapped_pt, after_vertex, _ = ctx
            if best is None or dist < best["dist"]:
                best = {
                    "feat_id": tid,
                    "sub_idx": si,
                    "after_vertex": after_vertex,
                    "snapped_pt": QgsPointXY(snapped_pt),
                    "dist": dist,
                }

    return best


def target_polyline(tfeat, sub_idx):
    """
    Return boundary sub-polyline for polygon or line feature.
    """
    tgeom = tfeat.geometry()
    if QgsWkbTypes.geometryType(tgeom.wkbType()) == QgsWkbTypes.PolygonGeometry:
        geoms = polygon_boundary_lines(tgeom)
        if 0 <= sub_idx < len(geoms):
            return polyline_from_geom(geoms[sub_idx])
        else:
            return None
    else:
        return polyline_from_geom(tgeom)


def build_path_on_polyline(polyline, infoA, ptA, infoB, ptB):
    """
    Build path on polyline between ptA and ptB using after_vertex indices.
    """
    if not polyline or len(polyline) < 2:
        return [ptA, ptB]

    idxA = max(0, min(infoA.get("after_vertex", 0), len(polyline)))
    idxB = max(0, min(infoB.get("after_vertex", 0), len(polyline)))

    path = [ptA]

    if idxA <= idxB:
        for k in range(idxA, idxB):
            if 0 <= k < len(polyline):
                path.append(QgsPointXY(polyline[k]))
    else:
        for k in range(idxA - 1, idxB - 1, -1):
            if 0 <= k < len(polyline):
                path.append(QgsPointXY(polyline[k]))

    path.append(ptB)

    # remove consecutive duplicates
    cleaned = []
    last_key = None
    for p in path:
        key = (p.x(), p.y())
        if key != last_key:
            cleaned.append(p)
            last_key = key

    if len(cleaned) < 2:
        return [ptA, ptB]
    return cleaned


def run_combined_water_boundary(valid_layer, lines_layer, tunnus):
    """
    Inlined version of CombinedWaterBoundaryProcessing.processAlgorithm(),
    but writing directly to memory layers instead of Processing sinks.
    """

    # --- parameters (you can tweak these) ---
    buffer_dist = 5.0      # water buffer distance
    merge_tol = 0.0        # endpoint snapping tolerance for dangle removal
    min_dangle = 12.0      # minimum dangling length to remove
    snap_tol = 0.001       # snap tolerance for interpolation

    boundary_source = valid_layer
    fields = boundary_source.fields()
    crs = boundary_source.crs()

    # create outputs
    adjusted_layer = QgsVectorLayer(
        f"LineString?crs={crs.authid()}",
        f"adjusted_segments_{tunnus}",
        "memory"
    )
    poly_layer = QgsVectorLayer(
        f"Polygon?crs={crs.authid()}",
        f"polygonized_areas_{tunnus}",
        "memory"
    )

    adj_pr = adjusted_layer.dataProvider()
    poly_pr = poly_layer.dataProvider()
    adj_pr.addAttributes(fields)
    poly_pr.addAttributes(fields)
    adjusted_layer.updateFields()
    poly_layer.updateFields()

    # === STEP 1: build water snap target from VALID_SEGMENTS + LINES ===

    # A) extract water segments
    water_res = processing.run(
        "native:extractbyattribute",
        {
            "INPUT": valid_layer,
            "FIELD": "segment_type",
            "OPERATOR": 0,  # '='
            "VALUE": "water",
            "OUTPUT": "memory:",
        },
    )
    water_layer = water_res["OUTPUT"]

    # B) buffer water segments (dissolve)
    buffer_res = processing.run(
        "native:buffer",
        {
            "INPUT": water_layer,
            "DISTANCE": buffer_dist,
            "SEGMENTS": 5,
            "END_CAP_STYLE": 0,
            "JOIN_STYLE": 0,
            "MITER_LIMIT": 2,
            "DISSOLVE": True,
            "OUTPUT": "memory:",
        },
    )
    buffer_layer = buffer_res["OUTPUT"]

    # C) clip LINES with water buffer
    clip_res = processing.run(
        "native:clip",
        {
            "INPUT": lines_layer,
            "OVERLAY": buffer_layer,
            "OUTPUT": "memory:",
        },
    )
    clipped_layer = clip_res["OUTPUT"]

    # D) build line graph + remove dangles on clipped_layer
    source = clipped_layer

    def node_key(pt):
        if merge_tol and merge_tol > 0:
            x = round(pt.x() / merge_tol) * merge_tol
            y = round(pt.y() / merge_tol) * merge_tol
        else:
            x = pt.x()
            y = pt.y()
        return (x, y)

    segments = []  # list of dicts: {points, length, k1, k2}
    nodes = defaultdict(list)

    for feat in source.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue

        if geom.isMultipart():
            lines = geom.asMultiPolyline()
        else:
            lines = [geom.asPolyline()]

        for pts in lines:
            if len(pts) < 2:
                continue

            part_geom = QgsGeometry.fromPolylineXY(pts)
            length = part_geom.length()

            seg_id = len(segments)
            k1 = node_key(pts[0])
            k2 = node_key(pts[-1])

            segments.append(
                {
                    "points": pts,
                    "length": length,
                    "k1": k1,
                    "k2": k2,
                }
            )

            nodes[k1].append(seg_id)
            nodes[k2].append(seg_id)

    crs_lines = source.crs()

    if not segments:
        # empty snap target
        merged_layer = QgsVectorLayer(
            f"LineString?crs={crs_lines.authid()}",
            "snap_target_empty",
            "memory"
        )
    else:
        # remove dangling segments
        remaining = set(range(len(segments)))
        changed = True

        while changed:
            changed = False

            # degrees
            deg = {}
            for nk, sids in nodes.items():
                c = sum(1 for sid in sids if sid in remaining)
                if c > 0:
                    deg[nk] = c

            for sid in list(remaining):
                seg = segments[sid]
                d1 = deg.get(seg["k1"], 0)
                d2 = deg.get(seg["k2"], 0)

                if seg["length"] < min_dangle and (d1 == 1 or d2 == 1):
                    remaining.remove(sid)
                    changed = True

        # put cleaned segments in memory layer
        mem_layer = QgsVectorLayer(
            f"LineString?crs={crs_lines.authid()}",
            "cleaned_segments",
            "memory"
        )
        mem_pr = mem_layer.dataProvider()
        new_feats = []
        for sid in sorted(remaining):
            seg = segments[sid]
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPolylineXY(seg["points"]))
            new_feats.append(f)
        mem_pr.addFeatures(new_feats)
        mem_layer.updateExtents()

        # dissolve + merge into snap target
        dissolve_res = processing.run(
            "native:dissolve",
            {"INPUT": mem_layer, "FIELD": [], "OUTPUT": "memory:"},
        )
        dissolved = dissolve_res["OUTPUT"]

        merge_res = processing.run(
            "native:mergelines",
            {"INPUT": dissolved, "OUTPUT": "memory:"},
        )
        merged_layer = merge_res["OUTPUT"]

    # === STEP 2: interpolate boundary, replace water, polygonize ===

    target_source = merged_layer

    # punkti_jnr field indices
    idx_pj_from = fields.indexFromName("punkti_jnr_from")
    idx_pj_to = fields.indexFromName("punkti_jnr_to")

    # decimals for rounding keys from snap tolerance
    if snap_tol > 0:
        decimals = max(0, int(math.ceil(-math.log10(snap_tol))) + 1)
    else:
        decimals = 8

    # build spatial index for snap target
    target_index = QgsSpatialIndex()
    target_feats = {}
    for f in target_source.getFeatures():
        target_index.insertFeature(f)
        target_feats[f.id()] = f

    # load boundary segments
    boundary_features = []
    geom_by_id = {}
    attrs_by_id = {}
    segtype_by_id = {}

    for f in boundary_source.getFeatures():
        boundary_features.append(f)
        geom_by_id[f.id()] = QgsGeometry(f.geometry())
        attrs_by_id[f.id()] = f.attributes()
        segtype_by_id[f.id()] = f["segment_type"]

    moved_points = {}
    RAY_LENGTH = 1e6

    # PASS 1: extend/shorten interpolate segments
    for feat in boundary_features:
        if segtype_by_id.get(feat.id()) != "interpolate":
            continue

        geom = geom_by_id[feat.id()]
        polyline = polyline_from_geom(geom)
        if not polyline or len(polyline) < 2:
            continue

        p_from = QgsPointXY(polyline[0])
        p_to = QgsPointXY(polyline[-1])

        pf = feat["piiripunkt_from"]
        pt = feat["piiripunkt_to"]

        movable_is_from = False
        movable_is_to = False

        if pf == "VEEKOGU_TELJE_PUNKT" and pt != "VEEKOGU_TELJE_PUNKT":
            movable_is_from = True
        elif pt == "VEEKOGU_TELJE_PUNKT" and pf != "VEEKOGU_TELJE_PUNKT":
            movable_is_to = True
        else:
            continue

        if movable_is_from:
            movable_pt = p_from
            fixed_pt = p_to
            movable_index = 0
        else:
            movable_pt = p_to
            fixed_pt = p_from
            movable_index = len(polyline) - 1

        dx = movable_pt.x() - fixed_pt.x()
        dy = movable_pt.y() - fixed_pt.y()
        length = math.hypot(dx, dy)
        if length == 0:
            continue

        ux = dx / length
        uy = dy / length

        far_pt = QgsPointXY(
            fixed_pt.x() + ux * RAY_LENGTH,
            fixed_pt.y() + uy * RAY_LENGTH,
        )

        ray_geom = QgsGeometry.fromPolylineXY([fixed_pt, far_pt])

        candidate_ids = target_index.intersects(ray_geom.boundingBox())
        best_point = None
        best_dist = None

        for tid in candidate_ids:
            tfeat = target_feats[tid]
            tgeom = tfeat.geometry()

            if QgsWkbTypes.geometryType(tgeom.wkbType()) == QgsWkbTypes.PolygonGeometry:
                tgeoms = polygon_boundary_lines(tgeom)
            else:
                tgeoms = [tgeom]

            for tg in tgeoms:
                inter = tg.intersection(ray_geom)
                if inter.isEmpty():
                    continue

                pts = get_points_from_geometry(inter)
                for ipt in pts:
                    dist = math.hypot(ipt.x() - fixed_pt.x(), ipt.y() - fixed_pt.y())
                    if best_point is None or dist < best_dist:
                        best_point = ipt
                        best_dist = dist

        if best_point is None:
            continue

        if best_dist <= snap_tol:
            continue

        new_polyline = list(polyline)
        new_polyline[movable_index] = best_point
        new_geom = QgsGeometry.fromPolylineXY(new_polyline)
        geom_by_id[feat.id()] = new_geom

        old_key = point_key(movable_pt, decimals)
        moved_points[old_key] = best_point

    # PASS 2: move connected water endpoints that share moved interpolate endpoints
    for feat in boundary_features:
        if segtype_by_id.get(feat.id()) != "water":
            continue

        geom = geom_by_id[feat.id()]
        polyline = polyline_from_geom(geom)
        if not polyline or len(polyline) < 2:
            continue

        p0 = QgsPointXY(polyline[0])
        p1 = QgsPointXY(polyline[-1])

        k0 = point_key(p0, decimals)
        k1 = point_key(p1, decimals)

        changed = False

        if k0 in moved_points:
            polyline[0] = moved_points[k0]
            changed = True

        if k1 in moved_points:
            polyline[-1] = moved_points[k1]
            changed = True

        if changed:
            geom_by_id[feat.id()] = QgsGeometry.fromPolylineXY(polyline)

    # PASS 3: replace water segments along snap target, write all adjusted lines
    fixed_keys = set()
    for new_pt in moved_points.values():
        fixed_keys.add(point_key(new_pt, decimals))

    line_geoms_for_polygonize = []
    out_line_feats = []

    for feat in boundary_features:
        segtype = segtype_by_id.get(feat.id())
        base_attrs = attrs_by_id[feat.id()]
        base_geom = geom_by_id[feat.id()]

        # non-water: just output adjusted geometry
        if segtype != "water":
            new_feat = QgsFeature()
            new_feat.setFields(fields)
            new_feat.setAttributes(base_attrs)
            new_feat.setGeometry(base_geom)
            out_line_feats.append(new_feat)
            line_geoms_for_polygonize.append(QgsGeometry(base_geom))
            continue

        polyline = polyline_from_geom(base_geom)
        if not polyline or len(polyline) < 2:
            new_feat = QgsFeature()
            new_feat.setFields(fields)
            new_feat.setAttributes(base_attrs)
            new_feat.setGeometry(base_geom)
            out_line_feats.append(new_feat)
            line_geoms_for_polygonize.append(QgsGeometry(base_geom))
            continue

        p_start = QgsPointXY(polyline[0])
        p_end = QgsPointXY(polyline[-1])

        k_start = point_key(p_start, decimals)
        k_end = point_key(p_end, decimals)

        start_fixed = k_start in fixed_keys
        end_fixed = k_end in fixed_keys

        info_start = closest_segment_info(p_start, target_index, target_feats)
        info_end = closest_segment_info(p_end, target_index, target_feats)

        if info_start is None or info_end is None:
            new_feat = QgsFeature()
            new_feat.setFields(fields)
            new_feat.setAttributes(base_attrs)
            new_feat.setGeometry(base_geom)
            out_line_feats.append(new_feat)
            line_geoms_for_polygonize.append(QgsGeometry(base_geom))
            continue

        snapped_start = info_start["snapped_pt"]
        snapped_end = info_end["snapped_pt"]

        final_start = p_start if start_fixed else snapped_start
        final_end = p_end if end_fixed else snapped_end

        path_points = None

        if (info_start["feat_id"] == info_end["feat_id"] and
                info_start["sub_idx"] == info_end["sub_idx"]):
            tfeat = target_feats[info_start["feat_id"]]
            polyline_target = target_polyline(tfeat, info_start["sub_idx"])
            if polyline_target and len(polyline_target) >= 2:
                path_boundary = build_path_on_polyline(
                    polyline_target,
                    info_start,
                    snapped_start,
                    info_end,
                    snapped_end,
                )
                path_points = list(path_boundary)
                path_points[0] = final_start
                path_points[-1] = final_end

        if path_points is None or len(path_points) < 2:
            path_points = [final_start, final_end]

        num_segments = max(1, len(path_points) - 1)

        base_jnr_from = None
        if idx_pj_from >= 0:
            try:
                base_jnr_from = int(base_attrs[idx_pj_from])
            except Exception:
                base_jnr_from = None

        if base_jnr_from is None:
            base_jnr_from = 0

        for i in range(num_segments):
            seg_start_pt = path_points[i]
            seg_end_pt = path_points[i + 1]

            new_attrs = list(base_attrs)

            if idx_pj_from >= 0:
                new_attrs[idx_pj_from] = base_jnr_from + i
            if idx_pj_to >= 0:
                new_attrs[idx_pj_to] = base_jnr_from + i + 1

            new_geom = QgsGeometry.fromPolylineXY([seg_start_pt, seg_end_pt])

            new_feat = QgsFeature()
            new_feat.setFields(fields)
            new_feat.setAttributes(new_attrs)
            new_feat.setGeometry(new_geom)
            out_line_feats.append(new_feat)
            line_geoms_for_polygonize.append(QgsGeometry(new_geom))

    # add all adjusted lines
    if out_line_feats:
        adj_pr.addFeatures(out_line_feats)
        adjusted_layer.updateExtents()

    # PASS 4: polygonize final lines -> polygons
    if line_geoms_for_polygonize:
        try:
            polygons_geom = QgsGeometry.polygonize(line_geoms_for_polygonize)
        except Exception:
            polygons_geom = None

        if polygons_geom is not None and not polygons_geom.isEmpty():
            poly_geoms_list = []

            gtype = QgsWkbTypes.geometryType(polygons_geom.wkbType())
            if gtype == QgsWkbTypes.PolygonGeometry:
                if QgsWkbTypes.isMultiType(polygons_geom.wkbType()):
                    for poly in polygons_geom.asMultiPolygon():
                        poly_geoms_list.append(QgsGeometry.fromPolygonXY(poly))
                else:
                    poly = polygons_geom.asPolygon()
                    if poly:
                        poly_geoms_list.append(QgsGeometry.fromPolygonXY(poly))
            else:
                try:
                    for g in polygons_geom.asGeometryCollection():
                        if QgsWkbTypes.geometryType(g.wkbType()) == QgsWkbTypes.PolygonGeometry:
                            if QgsWkbTypes.isMultiType(g.wkbType()):
                                for poly in g.asMultiPolygon():
                                    poly_geoms_list.append(QgsGeometry.fromPolygonXY(poly))
                            else:
                                poly = g.asPolygon()
                                if poly:
                                    poly_geoms_list.append(QgsGeometry.fromPolygonXY(poly))
                except Exception:
                    pass

            poly_feats = []
            for pg in poly_geoms_list:
                f = QgsFeature()
                f.setFields(fields)
                f.setAttributes([None] * len(fields))
                f.setGeometry(pg)
                poly_feats.append(f)

            if poly_feats:
                poly_pr.addFeatures(poly_feats)
                poly_layer.updateExtents()

    # add both layers to project
    #QgsProject.instance().addMapLayer(adjusted_layer) Comment out adjusted segments
    QgsProject.instance().addMapLayer(poly_layer)


# --------------------------- MAIN ACTION ------------------------------

# 1) tunnus from clicked feature
tunnus = '[% "tunnus" %]'.strip()

if not tunnus:
    iface.messageBar().pushMessage(
        "VALIS lõigud",
        "Tunnus puudub (veeru 'tunnus' väärtus on tühi).",
        level=Qgis.Warning,
        duration=5
    )
    raise Exception("Missing tunnus")

# 2) WFS request (ky_punktid_kehtiv)
base_url = "https://gsavalik.envir.ee/geoserver/kataster/wfs"
params = {
    "service": "WFS",
    "version": "1.0.0",
    "request": "GetFeature",
    "typeName": "kataster:ky_punktid_kehtiv",
    "outputFormat": "json",
    "CQL_FILTER": f"tunnus='{tunnus}'"
}
url = base_url + "?" + urllib.parse.urlencode(params)

try:
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception as e:
    iface.messageBar().pushMessage(
        "VALIS lõigud",
        f"Viga WFS päringul: {e}",
        level=Qgis.Critical,
        duration=8
    )
    raise

features_json = data.get("features", [])
if not features_json:
    iface.messageBar().pushMessage(
        "VALIS lõigud",
        f"Punkte ei leitud (tunnus {tunnus}).",
        level=Qgis.Info,
        duration=5
    )
    raise Exception("No features returned")

# 3) Collect VALIS points
points = []
for fjson in features_json:
    geom = fjson.get("geometry")
    if not geom or geom.get("type") != "Point":
        continue

    coords = geom.get("coordinates", None)
    if not coords or len(coords) < 2:
        continue

    x, y = coords[0], coords[1]
    pt = QgsPointXY(x, y)
    props = fjson.get("properties", {})

    if props.get("piiri_tyyp") != "VALIS":
        continue

    jnr_raw = props.get("punkti_jnr")
    try:
        jnr = int(jnr_raw)
    except Exception:
        jnr = jnr_raw

    points.append((jnr, props, pt))

if len(points) < 2:
    iface.messageBar().pushMessage(
        "VALIS lõigud",
        f"Leiti vähem kui 2 VALIS piiri punkti (tunnus {tunnus}), segmente ei loodud.",
        level=Qgis.Warning,
        duration=6
    )
    raise Exception("Not enough VALIS points")

points.sort(key=lambda x: x[0])

# 4) BBOX AROUND VALIS POINTS + BUFFER FOR ETAK
bbox = None
for _, _, pt in points:
    if bbox is None:
        bbox = QgsRectangle(pt, pt)
    else:
        bbox.include(pt)

buffer_dist_bbox = 200.0  # 200 m
bbox_buffered = QgsRectangle(bbox)
bbox_buffered.grow(buffer_dist_bbox)

minx = bbox_buffered.xMinimum()
miny = bbox_buffered.yMinimum()
maxx = bbox_buffered.xMaximum()
maxy = bbox_buffered.yMaximum()

# 5) Create valis_segments layer
sample_pp_nr = None
sample_p_jnr = None
sample_pp = None

for (_, props, _) in points:
    if sample_pp_nr is None:
        sample_pp_nr = props.get("piiripunkti_nr")
    if sample_p_jnr is None:
        sample_p_jnr = props.get("punkti_jnr")
    if sample_pp is None:
        sample_pp = props.get("piiripunkt")
    if sample_pp_nr is not None and sample_p_jnr is not None and sample_pp is not None:
        break

out_fields = QgsFields()
out_fields.append(QgsField("piiripunkti_nr_from", qvariant_type(sample_pp_nr)))
out_fields.append(QgsField("piiripunkti_nr_to",   qvariant_type(sample_pp_nr)))
out_fields.append(QgsField("punkti_jnr_from",     qvariant_type(sample_p_jnr)))
out_fields.append(QgsField("punkti_jnr_to",       qvariant_type(sample_p_jnr)))
out_fields.append(QgsField("piiripunkt_from",     qvariant_type(sample_pp)))
out_fields.append(QgsField("piiripunkt_to",       qvariant_type(sample_pp)))
out_fields.append(QgsField("segment_type",        QVariant.String))

project = QgsProject.instance()
layer_name = "valis_segments"

existing = project.mapLayersByName(layer_name)
if existing:
    layer = existing[0]
    provider = layer.dataProvider()
    ids = [f.id() for f in layer.getFeatures()]
    if ids:
        provider.deleteFeatures(ids)
else:
    layer = QgsVectorLayer("LineString?crs=EPSG:3301", layer_name, "memory")
    provider = layer.dataProvider()
    provider.addAttributes(out_fields)
    layer.updateFields()
    #project.addMapLayer(layer)   Kommenteerin välja segmenti kihi lisamise

fields = layer.fields()

# 6) Build segments (closed ring)
new_feats = []
total = len(points)

for i, (jnr_from, props_from, pt_from) in enumerate(points):
    jnr_to, props_to, pt_to = points[(i + 1) % total]

    piiripunkti_nr_from = props_from.get("piiripunkti_nr")
    piiripunkti_nr_to   = props_to.get("piiripunkti_nr")
    punkti_jnr_from     = props_from.get("punkti_jnr")
    punkti_jnr_to       = props_to.get("punkti_jnr")
    piiripunkt_from     = props_from.get("piiripunkt")
    piiripunkt_to       = props_to.get("piiripunkt")

    segment_type_val = _segment_type(piiripunkt_from, piiripunkt_to)

    feat = QgsFeature(fields)
    feat.setGeometry(QgsGeometry.fromPolylineXY([pt_from, pt_to]))
    feat.setAttributes([
        piiripunkti_nr_from,
        piiripunkti_nr_to,
        punkti_jnr_from,
        punkti_jnr_to,
        piiripunkt_from,
        piiripunkt_to,
        segment_type_val,
    ])
    new_feats.append(feat)

provider.addFeatures(new_feats)
layer.updateExtents()

# 7) Style valis_segments
symbol_base = QgsSymbol.defaultSymbol(layer.geometryType())
symbol_base.setWidth(0.5)

root_rule = QgsRuleBasedRenderer.Rule(None)

def make_rule(seg_value, color, label):
    sym = symbol_base.clone()
    sym.setColor(color)
    rule = QgsRuleBasedRenderer.Rule(
        sym,
        0, 0,
        f"\"segment_type\" = '{seg_value}'",
        label
    )
    root_rule.appendChild(rule)

make_rule("land",        QColor(0, 150, 0), "Land")
make_rule("water",       QColor(0, 0, 255), "Water")
make_rule("interpolate", QColor(255, 0, 0), "Interpolate")

renderer = QgsRuleBasedRenderer(root_rule)
layer.setRenderer(renderer)
layer.triggerRepaint()

# 8) ETAK WFS query -> vooluveekogu_telg
etak_base_url = "https://gsavalik.envir.ee/geoserver/etak/wfs"
etak_params = {
    "service": "WFS",
    "version": "1.1.0",
    "request": "GetFeature",
    "typeName": "etak:e_203_vooluveekogu_j",
    "srsName": "EPSG:3301",
    "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:3301",
    "outputFormat": "application/json",
}
etak_url = etak_base_url + "?" + urllib.parse.urlencode(etak_params)

etak_features = []
try:
    with urllib.request.urlopen(etak_url, timeout=30) as response:
        etak_data = json.loads(response.read().decode("utf-8"))
        etak_features = etak_data.get("features", [])
except Exception as e:
    iface.messageBar().pushMessage(
        "ETAK vooluveekogu",
        f"Viga ETAK WFS päringul: {e}",
        level=Qgis.Critical,
        duration=8
    )
    etak_features = []

etak_layer = None

if etak_features:
    etak_layer_name = "vooluveekogu_telg"
    existing_etak = project.mapLayersByName(etak_layer_name)
    if existing_etak:
        etak_layer = existing_etak[0]
        etak_provider = etak_layer.dataProvider()
        ids = [f.id() for f in etak_layer.getFeatures()]
        if ids:
            etak_provider.deleteFeatures(ids)
    else:
        sample_props = etak_features[0].get("properties", {})
        etak_fields = QgsFields()
        for key, val in sample_props.items():
            etak_fields.append(QgsField(key, qvariant_type(val)))

        etak_layer = QgsVectorLayer("LineString?crs=EPSG:3301", etak_layer_name, "memory")
        etak_provider = etak_layer.dataProvider()
        etak_provider.addAttributes(etak_fields)
        etak_layer.updateFields()
        #project.addMapLayer(etak_layer) Kommenteerin välja etak-i kihi välja kirjutamise

    etak_fields = etak_layer.fields()
    etak_new_feats = []

    for fjson in etak_features:
        geom_json = fjson.get("geometry")
        if not geom_json:
            continue

        geom_type = geom_json.get("type")
        coords = geom_json.get("coordinates", [])

        if geom_type == "LineString":
            lines = [coords]
        elif geom_type == "MultiLineString":
            lines = coords
        else:
            continue

        props = fjson.get("properties", {})
        attrs = [props.get(f.name()) for f in etak_fields]

        for line_coords in lines:
            pts = [QgsPointXY(x, y) for (x, y) in line_coords]
            geom = QgsGeometry.fromPolylineXY(pts)

            fet = QgsFeature(etak_fields)
            fet.setGeometry(geom)
            fet.setAttributes(attrs)
            etak_new_feats.append(fet)

    if etak_new_feats:
        etak_provider.addFeatures(etak_new_feats)
        etak_layer.updateExtents()

    # 9) run combined boundary / polygonization logic inlined
    run_combined_water_boundary(layer, etak_layer, tunnus)

    iface.messageBar().pushMessage(
        "VALIS + ETAK analüüs",
        f"Tunnus {tunnus}: {len(new_feats)} VALIS segmenti, ETAK vooluveekogu objekte: {len(etak_features)}. "
        "Loodi kohandatud segmendid ja polügoonid.",
        level=Qgis.Info,
        duration=10
    )

else:
    iface.messageBar().pushMessage(
        "ETAK vooluveekogu",
        f"Tunnus {tunnus}: ETAK vooluveekogu objekte ei leitud (bbox + 200 m). "
        "Loodi ainult valis_segments.",
        level=Qgis.Warning,
        duration=8
    )
