# -*- coding: utf-8 -*-
"""
Interpolate boundary extension + water segment replacement along snap target
and polygonize final boundaries into polygons.

QGIS 3.40 compatible (no geometry.boundary())
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterNumber,
    QgsFeature,
    QgsFeatureSink,
    QgsSpatialIndex,
    QgsWkbTypes,
    QgsGeometry,
    QgsPointXY,
)
import math


class InterpolateBoundaryExtension(QgsProcessingAlgorithm):

    PARAM_BOUNDARY = 'BOUNDARY'
    PARAM_SNAPTARGET = 'SNAPTARGET'
    PARAM_TOLERANCE = 'TOLERANCE'
    OUTPUT_LINES = 'OUTPUT_LINES'
    OUTPUT_POLYGONS = 'OUTPUT_POLYGONS'

    def tr(self, string):
        return QCoreApplication.translate('InterpolateBoundaryExtension', string)

    def createInstance(self):
        return InterpolateBoundaryExtension()

    def name(self):
        return 'interpolate_boundary_extension_replace_water_polygonize'

    def displayName(self):
        return self.tr('Interpolate boundary extension + replace water + polygonize')

    def group(self):
        return self.tr('Custom')

    def groupId(self):
        return 'custom'

    def shortHelpString(self):
        return self.tr(
            '1) Extends/shortens segments with segment_type="interpolate" until they hit '
            'a snap target (line/polygon boundary), moves connected "water" endpoints; '
            '2) replaces each water segment by new segment(s) that follow the snap target '
            'between its endpoints and recomputes punkti_jnr_from/to locally; '
            '3) polygonizes all resulting boundary segments into polygons.'
            'Tolerants on väikene kauguse lävi, millega skript otsustab, kui “lähedal” kaks punkti või lõik sihtjoonele juba on.'
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PARAM_BOUNDARY,
                self.tr('Boundary segments'),
                [QgsProcessing.TypeVectorLine]
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PARAM_SNAPTARGET,
                self.tr('Snap target (line or polygon)'),
                [QgsProcessing.TypeVectorLine, QgsProcessing.TypeVectorPolygon]
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_TOLERANCE,
                self.tr('Tolerance (map units)'),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.001,
                minValue=0.0
            )
        )

        # Väljund: lõigud
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_LINES,
                self.tr('Adjusted segments (lines)')
            )
        )

        # Väljund: polügoonid
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_POLYGONS,
                self.tr('Polygonized areas (from adjusted segments)')
            )
        )

    # -------------------- Abifunktsioonid --------------------

    @staticmethod
    def polygon_boundary_lines(poly_geom):
        """
        Tagastab kõik polügooni piiri LineString geomeetriad.
        Töötleb nii Simple kui MultiPolygon.
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

    @staticmethod
    def _polyline_from_geom(geom):
        if geom.isMultipart():
            m = geom.asMultiPolyline()
            return m[0] if m else None
        else:
            return geom.asPolyline()

    @staticmethod
    def _get_points_from_geometry(geom):
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

    @staticmethod
    def _point_key(pt, decimals):
        return (round(pt.x(), decimals), round(pt.y(), decimals))

    def _closest_segment_info(self, pt, target_index, target_feats):
        """
        Leia lähim segment snap-targetil ja tagasta:
        {
          'feat_id': target feature id,
          'sub_idx': indeks boundary-lis (polügooni puhul), muidu 0,
          'after_vertex': after_vertex indeks,
          'snapped_pt': QgsPointXY,
          'dist': float
        }
        Või None, kui ei leitud.
        """
        cand_ids = target_index.nearestNeighbor(pt, 10)
        if not cand_ids:
            return None

        best = None

        for tid in cand_ids:
            tfeat = target_feats[tid]
            tgeom = tfeat.geometry()

            if QgsWkbTypes.geometryType(tgeom.wkbType()) == QgsWkbTypes.PolygonGeometry:
                subgeoms = self.polygon_boundary_lines(tgeom)
            else:
                subgeoms = [tgeom]

            for si, sg in enumerate(subgeoms):
                ctx = sg.closestSegmentWithContext(pt)
                if ctx is None:
                    continue
                dist, snapped_pt, after_vertex, _ = ctx
                if best is None or dist < best['dist']:
                    best = {
                        'feat_id': tid,
                        'sub_idx': si,
                        'after_vertex': after_vertex,
                        'snapped_pt': QgsPointXY(snapped_pt),
                        'dist': dist
                    }

        return best

    def _target_polyline(self, tfeat, sub_idx):
        """
        Tagasta boundary-alampolüliini punktide list.
        """
        tgeom = tfeat.geometry()
        if QgsWkbTypes.geometryType(tgeom.wkbType()) == QgsWkbTypes.PolygonGeometry:
            geoms = self.polygon_boundary_lines(tgeom)
            if 0 <= sub_idx < len(geoms):
                return self._polyline_from_geom(geoms[sub_idx])
            else:
                return None
        else:
            return self._polyline_from_geom(tgeom)

    def _build_path_on_polyline(self, polyline, infoA, ptA, infoB, ptB):
        """
        Ehita tee polüliinil ptA ja ptB vahel, kasutades after_vertex indekseid.
        polyline: list[QgsPointXY]
        infoA/B: dict with 'after_vertex'
        ptA/ptB: snapped points (boundary-l).
        """
        if not polyline or len(polyline) < 2:
            return [ptA, ptB]

        idxA = max(0, min(infoA.get('after_vertex', 0), len(polyline)))
        idxB = max(0, min(infoB.get('after_vertex', 0), len(polyline)))

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

        # eemalda järjestikused duplikaadid
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

    # ----------------- Pea-masinavärk ------------------------

    def processAlgorithm(self, parameters, context, feedback):
        boundary_source = self.parameterAsSource(parameters, self.PARAM_BOUNDARY, context)
        target_source = self.parameterAsSource(parameters, self.PARAM_SNAPTARGET, context)
        tolerance = self.parameterAsDouble(parameters, self.PARAM_TOLERANCE, context)

        fields = boundary_source.fields()

        # Line sink
        (sink_lines, lines_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_LINES,
            context,
            fields,
            boundary_source.wkbType(),
            boundary_source.sourceCrs()
        )

        # Polygon sink (kasutame samu välju; väärtused jäävad vaikimisi NULL)
        (sink_polys, polys_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_POLYGONS,
            context,
            fields,
            QgsWkbTypes.Polygon,
            boundary_source.sourceCrs()
        )

        # punkti_jnr väljade indeksid
        idx_pj_from = fields.indexFromName('punkti_jnr_from')
        idx_pj_to   = fields.indexFromName('punkti_jnr_to')

        # decimals for rounding
        if tolerance > 0:
            decimals = max(0, int(math.ceil(-math.log10(tolerance))) + 1)
        else:
            decimals = 8

        # ------------------ Snap target index ------------------

        feedback.pushInfo(self.tr('Building spatial index for snap target...'))
        target_index = QgsSpatialIndex()
        target_feats = {}

        for f in target_source.getFeatures():
            target_index.insertFeature(f)
            target_feats[f.id()] = f

        # ------------------ Load boundary -----------------------

        feedback.pushInfo(self.tr('Reading boundary segments...'))

        boundary_features = []
        geom_by_id = {}
        attrs_by_id = {}
        segtype_by_id = {}

        for f in boundary_source.getFeatures():
            boundary_features.append(f)
            geom_by_id[f.id()] = QgsGeometry(f.geometry())
            attrs_by_id[f.id()] = f.attributes()
            segtype_by_id[f.id()] = f['segment_type']

        moved_points = {}  # vana tipp -> uus tipp (interpolate liigutatud otsad)

        RAY_LENGTH = 1e6

        # ------------------ PASS 1: Interpolate -----------------

        feedback.pushInfo(self.tr('Pass 1: interpolate segments (extend/shorten)...'))

        for feat in boundary_features:

            if segtype_by_id.get(feat.id()) != 'interpolate':
                continue

            geom = geom_by_id[feat.id()]
            polyline = self._polyline_from_geom(geom)
            if not polyline or len(polyline) < 2:
                continue

            p_from = QgsPointXY(polyline[0])
            p_to   = QgsPointXY(polyline[-1])

            pf = feat['piiripunkt_from']
            pt = feat['piiripunkt_to']

            movable_is_from = False
            movable_is_to = False

            if pf == 'VEEKOGU_TELJE_PUNKT' and pt != 'VEEKOGU_TELJE_PUNKT':
                movable_is_from = True
            elif pt == 'VEEKOGU_TELJE_PUNKT' and pf != 'VEEKOGU_TELJE_PUNKT':
                movable_is_to = True
            else:
                continue  # see interpolate lõik ei lähe siia reeglisse

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
                fixed_pt.y() + uy * RAY_LENGTH
            )

            ray_geom = QgsGeometry.fromPolylineXY([fixed_pt, far_pt])

            candidate_ids = target_index.intersects(ray_geom.boundingBox())

            best_point = None
            best_dist = None

            for tid in candidate_ids:
                tfeat = target_feats[tid]
                tgeom = tfeat.geometry()

                if QgsWkbTypes.geometryType(tgeom.wkbType()) == QgsWkbTypes.PolygonGeometry:
                    tgeoms = self.polygon_boundary_lines(tgeom)
                else:
                    tgeoms = [tgeom]

                for tg in tgeoms:
                    inter = tg.intersection(ray_geom)
                    if inter.isEmpty():
                        continue

                    pts = self._get_points_from_geometry(inter)
                    for ipt in pts:
                        dist = math.hypot(ipt.x() - fixed_pt.x(), ipt.y() - fixed_pt.y())
                        if best_point is None or dist < best_dist:
                            best_point = ipt
                            best_dist = dist

            if best_point is None:
                continue

            if best_dist <= tolerance:
                # juba piisavalt lähedal
                continue

            new_polyline = list(polyline)
            new_polyline[movable_index] = best_point
            new_geom = QgsGeometry.fromPolylineXY(new_polyline)
            geom_by_id[feat.id()] = new_geom

            old_key = self._point_key(movable_pt, decimals)
            moved_points[old_key] = best_point

        # ------------------ PASS 2: Water endpoints follow moved interpolate ----

        feedback.pushInfo(self.tr('Pass 2: moving connected water endpoints with interpolate...'))

        for feat in boundary_features:

            if segtype_by_id.get(feat.id()) != 'water':
                continue

            geom = geom_by_id[feat.id()]
            polyline = self._polyline_from_geom(geom)
            if not polyline or len(polyline) < 2:
                continue

            p0 = QgsPointXY(polyline[0])
            p1 = QgsPointXY(polyline[-1])

            k0 = self._point_key(p0, decimals)
            k1 = self._point_key(p1, decimals)

            changed = False

            if k0 in moved_points:
                polyline[0] = moved_points[k0]
                changed = True

            if k1 in moved_points:
                polyline[-1] = moved_points[k1]
                changed = True

            if changed:
                geom_by_id[feat.id()] = QgsGeometry.fromPolylineXY(polyline)

        # ------------------ PASS 3: Replace water segments along snap target -----

        feedback.pushInfo(self.tr('Pass 3: replacing water segments along snap target...'))

        # lukus tipud – neid enam ei nihutata (reegel 1 tulemus)
        fixed_keys = set()
        for new_pt in moved_points.values():
            fixed_keys.add(self._point_key(new_pt, decimals))

        # siia kogume kõik LÕPLIKUD joonegeomeetriad polügoniseerimiseks
        line_geoms_for_polygonize = []

        for feat in boundary_features:

            segtype = segtype_by_id.get(feat.id())
            base_attrs = attrs_by_id[feat.id()]
            base_geom = geom_by_id[feat.id()]

            # Kõik MITTE-water lõigud kirjutame lihtsalt välja (koos reegel 1 muudatustega)
            if segtype != 'water':
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(base_attrs)
                new_feat.setGeometry(base_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(base_geom))
                continue

            # Nüüd töötleme water lõiku: asendame selle uute lõikudega
            polyline = self._polyline_from_geom(base_geom)
            if not polyline or len(polyline) < 2:
                # mingi veider geomeetria, väljasta muutmata kujul
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(base_attrs)
                new_feat.setGeometry(base_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(base_geom))
                continue

            p_start = QgsPointXY(polyline[0])
            p_end   = QgsPointXY(polyline[-1])

            k_start = self._point_key(p_start, decimals)
            k_end   = self._point_key(p_end, decimals)

            start_fixed = k_start in fixed_keys
            end_fixed   = k_end in fixed_keys

            # Lähimad segmendid snap-targetil (liikmesuse jaoks + vaba otsa snappimiseks)
            info_start = self._closest_segment_info(p_start, target_index, target_feats)
            info_end   = self._closest_segment_info(p_end,   target_index, target_feats)

            # Kui ei leitud, ei ole meil midagi mõistlikku teha -> väljasta lõik muutmata
            if info_start is None or info_end is None:
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(base_attrs)
                new_feat.setGeometry(base_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(base_geom))
                continue

            # snapped punktid sihtjoonel
            snapped_start = info_start['snapped_pt']
            snapped_end   = info_end['snapped_pt']

            # Lõplikud otsad geomeetrias:
            #  - kui tipp on "fixed", kasuta olemasolevat koordinatti
            #  - kui vaba, kasuta snäpitud koordinatti
            final_start = p_start if start_fixed else snapped_start
            final_end   = p_end   if end_fixed   else snapped_end

            # Proovime teekonda mööda sama sihtjoont (sama feature + sama sub_idx)
            path_points = None

            if (info_start['feat_id'] == info_end['feat_id'] and
                    info_start['sub_idx'] == info_end['sub_idx']):
                tfeat = target_feats[info_start['feat_id']]
                polyline_target = self._target_polyline(tfeat, info_start['sub_idx'])
                if polyline_target and len(polyline_target) >= 2:
                    # Tee boundary-l: A'..vahevertexid..B'
                    path_boundary = self._build_path_on_polyline(
                        polyline_target,
                        info_start,
                        snapped_start,
                        info_end,
                        snapped_end
                    )
                    # Asenda algus/lõpp vastavalt fixed staatusele
                    path_points = list(path_boundary)
                    path_points[0] = final_start
                    path_points[-1] = final_end

            # Kui ei suutnud sihtjoonelt midagi loogilist ehitada, tee üks sirge lõik
            if path_points is None or len(path_points) < 2:
                path_points = [final_start, final_end]

            # Nüüd path_points määrab, kuidas water piirdub jooksma
            num_segments = max(1, len(path_points) - 1)

            # Arvutame lokaalset punkti_jnr jada
            base_jnr_from = None
            if idx_pj_from >= 0:
                try:
                    base_jnr_from = int(base_attrs[idx_pj_from])
                except Exception:
                    base_jnr_from = None

            if base_jnr_from is None:
                base_jnr_from = 0

            # Loome uued lõigud: jnr_from = base_jnr_from + i, jnr_to = base_jnr_from + i + 1
            for i in range(num_segments):
                seg_start_pt = path_points[i]
                seg_end_pt   = path_points[i + 1]

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
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(new_geom))

        # ------------------ PASS 4: Polygonize all final lines -----------------

        feedback.pushInfo(self.tr('Pass 4: polygonizing adjusted segments...'))

        polygons_geom = None
        if line_geoms_for_polygonize:
            try:
                polygons_geom = QgsGeometry.polygonize(line_geoms_for_polygonize)
            except Exception as e:
                feedback.pushInfo(self.tr('Polygonize failed: {}').format(str(e)))
                polygons_geom = None

        if polygons_geom is not None and not polygons_geom.isEmpty():
            gtype = QgsWkbTypes.geometryType(polygons_geom.wkbType())

            poly_geoms_list = []

            if gtype == QgsWkbTypes.PolygonGeometry:
                if QgsWkbTypes.isMultiType(polygons_geom.wkbType()):
                    # MultiPolygon
                    for poly in polygons_geom.asMultiPolygon():
                        poly_geoms_list.append(QgsGeometry.fromPolygonXY(poly))
                else:
                    # Single polygon
                    poly = polygons_geom.asPolygon()
                    if poly:
                        poly_geoms_list.append(QgsGeometry.fromPolygonXY(poly))
            else:
                # Võib olla GeometryCollection; proovime sealt polügoone välja võtta
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

            # Kirjutame polügoonid väljundisse
            for pg in poly_geoms_list:
                f = QgsFeature()
                f.setFields(fields)
                # kõik atribuudid jäävad vaikimisi NULL-iks (sobib struktuuri hoidmiseks)
                f.setAttributes([None] * len(fields))
                f.setGeometry(pg)
                sink_polys.addFeature(f, QgsFeatureSink.FastInsert)

        return {
            self.OUTPUT_LINES: lines_id,
            self.OUTPUT_POLYGONS: polys_id
        }
