# -*- coding: utf-8 -*-
"""
Combined algorithm:
1) Merge lines in water buffer & remove dangles (from VALID_SEGMENTS + LINES)
2) Use merged result as snap target for:
   Interpolate boundary extension + replace water + polygonize (on VALID_SEGMENTS)

Outputs:
  - Adjusted segments (lines)
  - Polygonized areas
"""

import math
from collections import defaultdict

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsGeometry,
    QgsPointXY,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterDistance,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsSpatialIndex,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis import processing


class CombinedWaterBoundaryProcessing(QgsProcessingAlgorithm):
    """
    Step 1:
      - From VALID_SEGMENTS, keep only segment_type = 'water'
      - Buffer by configurable distance (default 5 m, dissolve)
      - Clip LINES with this buffer
      - Remove short dangling branches
      - Dissolve + merge lines -> SNAP TARGET

    Step 2:
      - Use VALID_SEGMENTS as boundary segments
      - Use SNAP TARGET as 'snap target' for interpolation
      - Extend/shorten 'interpolate' segments, move connected 'water' endpoints
      - Replace water segments along snap target
      - Polygonize final boundaries
    """

    # Parameters
    PARAM_VALID_SEGMENTS = "VALID_SEGMENTS"
    PARAM_LINES = "LINES"
    PARAM_BUFFER_DIST = "BUFFER_DIST"
    PARAM_MERGE_TOL = "MERGE_TOL"
    PARAM_MIN_DANGLE = "MIN_DANGLE"
    PARAM_SNAP_TOL = "SNAP_TOL"

    OUTPUT_LINES = "OUTPUT_LINES"
    OUTPUT_POLYGONS = "OUTPUT_POLYGONS"

    # ------------------------------------------------------------------ #
    # Standard metadata
    # ------------------------------------------------------------------ #
    def tr(self, string):
        return QCoreApplication.translate("CombinedWaterBoundaryProcessing", string)

    def createInstance(self):
        return CombinedWaterBoundaryProcessing()

    def name(self):
        return "combined_water_merge_interpolate_polygonize"

    def displayName(self):
        return self.tr(
            "Piirid veekogu teljele"
        )

    def group(self):
        return self.tr("Custom")

    def groupId(self):
        return "custom"

    def shortHelpString(self):
        return self.tr(
            "1) From VALID_SEGMENTS, keeps only segment_type='water', buffers them by "
            "a configurable distance (default 5 m) and dissolves.\n"
            "2) Clips the LINES layer with this buffer, removes short dangling branches "
            "(based on merge tolerance and minimum dangle length), and merges remaining "
            "lines into long LineString(s). This becomes the SNAP TARGET.\n\n"
            "3) Uses VALID_SEGMENTS as boundary segments and the merged SNAP TARGET as "
            "the snap target for:\n"
            "   • Extending/shortening segments with segment_type='interpolate' until "
            "     they hit the snap target, moving connected 'water' endpoints.\n"
            "   • Replacing each 'water' segment by new segment(s) that follow the "
            "     snap target between its endpoints, recomputing punkti_jnr_from/to locally.\n"
            "   • Polygonizing all resulting boundary segments into polygons.\n\n"
            "Water buffer distance: radius used to buffer water segments.\n"
            "Merge tolerance: snapping tolerance for endpoint grouping in the dangle-removal graph.\n"
            "Min dangle length: any line with at least one degree-1 endpoint and shorter "
            "than this length is removed (possibly in several passes).\n"
            "Snap tolerance: distance threshold used in interpolation/replacement to decide "
            "when points or segments are already close enough to the target."
        )

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    def initAlgorithm(self, config=None):
        # 1) Valid segments layer
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PARAM_VALID_SEGMENTS,
                self.tr("Valis segments (boundary, segment_type/water/interpolate)"),
                [QgsProcessing.TypeVectorLine],
            )
        )

        # 2) Lines to be clipped/merged to form water snap target
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.PARAM_LINES,
                self.tr("Water lines (etak / vooluveekogu)"),
                [QgsProcessing.TypeVectorLine],
            )
        )

        # Water buffer distance
        self.addParameter(
            QgsProcessingParameterDistance(
                self.PARAM_BUFFER_DIST,
                self.tr("Water buffer distance"),
                defaultValue=5.0,
                parentParameterName=self.PARAM_VALID_SEGMENTS,
            )
        )

        # Endpoint snapping tolerance for dangle-removal graph
        self.addParameter(
            QgsProcessingParameterDistance(
                self.PARAM_MERGE_TOL,
                self.tr("Endpoint snapping tolerance (for dangle removal)"),
                defaultValue=0.0,
                parentParameterName=self.PARAM_LINES,
            )
        )

        # Minimum dangling length to remove
        self.addParameter(
            QgsProcessingParameterDistance(
                self.PARAM_MIN_DANGLE,
                self.tr("Minimum dangling length to remove"),
                defaultValue=12.0,
                parentParameterName=self.PARAM_LINES,
            )
        )

        # Snap tolerance for interpolation/replacement
        self.addParameter(
            QgsProcessingParameterNumber(
                self.PARAM_SNAP_TOL,
                self.tr("Snap tolerance for interpolation (map units)"),
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.001,
                minValue=0.0,
            )
        )

        # Outputs (from interpolation + polygonize)
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_LINES,
                self.tr("Adjusted segments (lines)"),
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT_POLYGONS,
                self.tr("Polygonized areas (from adjusted segments)"),
            )
        )

    # ------------------------------------------------------------------ #
    # Helper methods from the interpolation script
    # ------------------------------------------------------------------ #
    @staticmethod
    def polygon_boundary_lines(poly_geom):
        """
        Return all polygon boundary LineString geometries.
        Works for single and multi polygons.
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
        Find closest segment on snap target and return:
        {
          'feat_id': target feature id,
          'sub_idx': boundary index (for polygon), else 0,
          'after_vertex': after_vertex index,
          'snapped_pt': QgsPointXY,
          'dist': float
        }
        or None.
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
                if best is None or dist < best["dist"]:
                    best = {
                        "feat_id": tid,
                        "sub_idx": si,
                        "after_vertex": after_vertex,
                        "snapped_pt": QgsPointXY(snapped_pt),
                        "dist": dist,
                    }

        return best

    def _target_polyline(self, tfeat, sub_idx):
        """
        Return boundary sub-polyline for polygon or line feature.
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

    # ------------------------------------------------------------------ #
    # Core algorithm
    # ------------------------------------------------------------------ #
    def processAlgorithm(self, parameters, context, feedback):
        # INPUTS
        valid_layer = self.parameterAsVectorLayer(
            parameters, self.PARAM_VALID_SEGMENTS, context
        )
        lines_layer = self.parameterAsVectorLayer(
            parameters, self.PARAM_LINES, context
        )

        if valid_layer is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.PARAM_VALID_SEGMENTS)
            )
        if lines_layer is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.PARAM_LINES)
            )

        buffer_dist = self.parameterAsDouble(
            parameters, self.PARAM_BUFFER_DIST, context
        )
        merge_tol = self.parameterAsDouble(parameters, self.PARAM_MERGE_TOL, context)
        min_dangle = self.parameterAsDouble(parameters, self.PARAM_MIN_DANGLE, context)
        snap_tol = self.parameterAsDouble(parameters, self.PARAM_SNAP_TOL, context)

        # OUTPUT sinks (for interpolation + polygonize)
        boundary_source = self.parameterAsSource(
            parameters, self.PARAM_VALID_SEGMENTS, context
        )
        fields = boundary_source.fields()

        (sink_lines, lines_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_LINES,
            context,
            fields,
            boundary_source.wkbType(),
            boundary_source.sourceCrs(),
        )

        (sink_polys, polys_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT_POLYGONS,
            context,
            fields,
            QgsWkbTypes.Polygon,
            boundary_source.sourceCrs(),
        )

        if feedback.isCanceled():
            return {}

        # ==================================================================
        # STEP 1: Merge lines in water buffer & remove dangles
        # ==================================================================
        feedback.pushInfo("STEP 1: Building water buffer and merging lines…")

        # A. Filter VALID_SEGMENTS where segment_type = 'water'
        feedback.pushInfo("  Filtering VALID_SEGMENTS where segment_type = 'water'…")
        water_res = processing.run(
            "native:extractbyattribute",
            {
                "INPUT": valid_layer,
                "FIELD": "segment_type",
                "OPERATOR": 0,  # '='
                "VALUE": "water",
                "OUTPUT": "memory:",
            },
            context=context,
            feedback=feedback,
        )
        water_layer = water_res["OUTPUT"]

        # B. Buffer water segments by configurable distance (dissolve)
        feedback.pushInfo(
            f"  Buffering water segments by {buffer_dist} (dissolve)…"
        )
        buffer_res = processing.run(
            "native:buffer",
            {
                "INPUT": water_layer,
                "DISTANCE": buffer_dist,
                "SEGMENTS": 5,
                "END_CAP_STYLE": 0,  # round
                "JOIN_STYLE": 0,  # round
                "MITER_LIMIT": 2,
                "DISSOLVE": True,
                "OUTPUT": "memory:",
            },
            context=context,
            feedback=feedback,
        )
        buffer_layer = buffer_res["OUTPUT"]

        # C. Clip LINES layer with water buffer
        feedback.pushInfo("  Clipping LINES with water buffer…")
        clip_res = processing.run(
            "native:clip",
            {
                "INPUT": lines_layer,
                "OVERLAY": buffer_layer,
                "OUTPUT": "memory:",
            },
            context=context,
            feedback=feedback,
        )
        clipped_layer = clip_res["OUTPUT"]

        # From here, reuse the line-graph + dangle-removal logic on clipped_layer
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
        nodes = defaultdict(list)  # node_key -> list of segment indices

        feedback.pushInfo("  Building node graph from clipped lines…")

        for feat in source.getFeatures():
            if feedback.isCanceled():
                break

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

        crs = source.sourceCrs()
        merged_layer = None

        if not segments:
            feedback.pushInfo(
                "  No line segments found after clipping. Using empty snap target."
            )
            # Empty snap target layer
            merged_layer = QgsVectorLayer(
                "LineString?crs=" + crs.authid(), "snap_target_empty", "memory"
            )
        else:
            # Remove dangling segments
            feedback.pushInfo("  Removing dangling segments…")
            remaining = set(range(len(segments)))
            changed = True

            while changed and not feedback.isCanceled():
                changed = False

                # Compute degrees
                deg = {}
                for nk, sids in nodes.items():
                    c = sum(1 for sid in sids if sid in remaining)
                    if c > 0:
                        deg[nk] = c

                # Peel off short danglers
                for sid in list(remaining):
                    seg = segments[sid]
                    d1 = deg.get(seg["k1"], 0)
                    d2 = deg.get(seg["k2"], 0)

                    if seg["length"] < min_dangle and (d1 == 1 or d2 == 1):
                        remaining.remove(sid)
                        changed = True

            feedback.pushInfo(
                f"  Remaining segments after dangle removal: {len(remaining)}"
            )

            # Put cleaned segments into memory layer
            mem_layer = QgsVectorLayer(
                "LineString?crs=" + crs.authid(), "cleaned_segments", "memory"
            )
            mem_pr = mem_layer.dataProvider()
            mem_layer.startEditing()

            new_feats = []
            for sid in sorted(remaining):
                seg = segments[sid]
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPolylineXY(seg["points"]))
                new_feats.append(f)

            mem_pr.addFeatures(new_feats)
            mem_layer.commitChanges()
            mem_layer.updateExtents()

            # Dissolve + merge
            feedback.pushInfo("  Dissolving and merging lines for snap target…")
            dissolve_res = processing.run(
                "native:dissolve",
                {
                    "INPUT": mem_layer,
                    "FIELD": [],
                    "OUTPUT": "memory:",
                },
                context=context,
                feedback=feedback,
            )
            dissolved = dissolve_res["OUTPUT"]

            merge_res = processing.run(
                "native:mergelines",
                {
                    "INPUT": dissolved,
                    "OUTPUT": "memory:",
                },
                context=context,
                feedback=feedback,
            )
            merged_layer = merge_res["OUTPUT"]

        if feedback.isCanceled():
            return {}

        # ==================================================================
        # STEP 2: Interpolate boundary extension + replace water + polygonize
        # ==================================================================
        feedback.pushInfo(
            "STEP 2: Interpolating boundary, replacing water segments, polygonizing…"
        )

        target_source = merged_layer

        # punkti_jnr field indices
        idx_pj_from = fields.indexFromName("punkti_jnr_from")
        idx_pj_to = fields.indexFromName("punkti_jnr_to")

        # decimals for rounding (snap tolerance)
        if snap_tol > 0:
            decimals = max(0, int(math.ceil(-math.log10(snap_tol))) + 1)
        else:
            decimals = 8

        # ---------- Build spatial index for snap target ----------
        feedback.pushInfo("  Building spatial index for snap target…")
        target_index = QgsSpatialIndex()
        target_feats = {}

        for f in target_source.getFeatures():
            target_index.insertFeature(f)
            target_feats[f.id()] = f

        # ---------- Load boundary segments ----------
        feedback.pushInfo("  Reading boundary segments…")

        boundary_features = []
        geom_by_id = {}
        attrs_by_id = {}
        segtype_by_id = {}

        for f in boundary_source.getFeatures():
            boundary_features.append(f)
            geom_by_id[f.id()] = QgsGeometry(f.geometry())
            attrs_by_id[f.id()] = f.attributes()
            segtype_by_id[f.id()] = f["segment_type"]

        moved_points = {}  # old vertex -> new vertex (for moved interpolate endpoints)
        RAY_LENGTH = 1e6

        # ---------- PASS 1: Interpolate (extend/shorten 'interpolate' segments) ----------
        feedback.pushInfo("  Pass 1: interpolate segments (extend/shorten)…")

        for feat in boundary_features:
            if segtype_by_id.get(feat.id()) != "interpolate":
                continue

            geom = geom_by_id[feat.id()]
            polyline = self._polyline_from_geom(geom)
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
                continue  # this interpolate segment not handled by rule

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

                if (
                    QgsWkbTypes.geometryType(tgeom.wkbType())
                    == QgsWkbTypes.PolygonGeometry
                ):
                    tgeoms = self.polygon_boundary_lines(tgeom)
                else:
                    tgeoms = [tgeom]

                for tg in tgeoms:
                    inter = tg.intersection(ray_geom)
                    if inter.isEmpty():
                        continue

                    pts = self._get_points_from_geometry(inter)
                    for ipt in pts:
                        dist = math.hypot(
                            ipt.x() - fixed_pt.x(), ipt.y() - fixed_pt.y()
                        )
                        if best_point is None or dist < best_dist:
                            best_point = ipt
                            best_dist = dist

            if best_point is None:
                continue

            if best_dist <= snap_tol:
                # already close enough
                continue

            new_polyline = list(polyline)
            new_polyline[movable_index] = best_point
            new_geom = QgsGeometry.fromPolylineXY(new_polyline)
            geom_by_id[feat.id()] = new_geom

            old_key = self._point_key(movable_pt, decimals)
            moved_points[old_key] = best_point

        # ---------- PASS 2: Water endpoints follow moved interpolate endpoints ----------
        feedback.pushInfo(
            "  Pass 2: moving connected water endpoints with interpolate…"
        )

        for feat in boundary_features:
            if segtype_by_id.get(feat.id()) != "water":
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

        # ---------- PASS 3: Replace water segments along snap target ----------
        feedback.pushInfo("  Pass 3: replacing water segments along snap target…")

        fixed_keys = set()
        for new_pt in moved_points.values():
            fixed_keys.add(self._point_key(new_pt, decimals))

        line_geoms_for_polygonize = []

        for feat in boundary_features:
            segtype = segtype_by_id.get(feat.id())
            base_attrs = attrs_by_id[feat.id()]
            base_geom = geom_by_id[feat.id()]

            # Non-water segments: just write out with modified geometry
            if segtype != "water":
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(base_attrs)
                new_feat.setGeometry(base_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(base_geom))
                continue

            polyline = self._polyline_from_geom(base_geom)
            if not polyline or len(polyline) < 2:
                # weird geometry, output unchanged
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(base_attrs)
                new_feat.setGeometry(base_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(base_geom))
                continue

            p_start = QgsPointXY(polyline[0])
            p_end = QgsPointXY(polyline[-1])

            k_start = self._point_key(p_start, decimals)
            k_end = self._point_key(p_end, decimals)

            start_fixed = k_start in fixed_keys
            end_fixed = k_end in fixed_keys

            info_start = self._closest_segment_info(
                p_start, target_index, target_feats
            )
            info_end = self._closest_segment_info(p_end, target_index, target_feats)

            # If no suitable snap target found, keep original geometry
            if info_start is None or info_end is None:
                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(base_attrs)
                new_feat.setGeometry(base_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
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
                polyline_target = self._target_polyline(tfeat, info_start["sub_idx"])
                if polyline_target and len(polyline_target) >= 2:
                    path_boundary = self._build_path_on_polyline(
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

                new_geom = QgsGeometry.fromPolylineXY(
                    [seg_start_pt, seg_end_pt]
                )

                new_feat = QgsFeature()
                new_feat.setFields(fields)
                new_feat.setAttributes(new_attrs)
                new_feat.setGeometry(new_geom)
                sink_lines.addFeature(new_feat, QgsFeatureSink.FastInsert)
                line_geoms_for_polygonize.append(QgsGeometry(new_geom))

        # ---------- PASS 4: Polygonize all final lines ----------
        feedback.pushInfo("  Pass 4: polygonizing adjusted segments…")

        polygons_geom = None
        if line_geoms_for_polygonize:
            try:
                polygons_geom = QgsGeometry.polygonize(line_geoms_for_polygonize)
            except Exception as e:
                feedback.pushInfo("  Polygonize failed: {}".format(str(e)))
                polygons_geom = None

        if polygons_geom is not None and not polygons_geom.isEmpty():
            gtype = QgsWkbTypes.geometryType(polygons_geom.wkbType())
            poly_geoms_list = []

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
                        if (
                            QgsWkbTypes.geometryType(g.wkbType())
                            == QgsWkbTypes.PolygonGeometry
                        ):
                            if QgsWkbTypes.isMultiType(g.wkbType()):
                                for poly in g.asMultiPolygon():
                                    poly_geoms_list.append(
                                        QgsGeometry.fromPolygonXY(poly)
                                    )
                            else:
                                poly = g.asPolygon()
                                if poly:
                                    poly_geoms_list.append(
                                        QgsGeometry.fromPolygonXY(poly)
                                    )
                except Exception:
                    pass

            for pg in poly_geoms_list:
                f = QgsFeature()
                f.setFields(fields)
                # attributes left NULL (just schema preserved)
                f.setAttributes([None] * len(fields))
                f.setGeometry(pg)
                sink_polys.addFeature(f, QgsFeatureSink.FastInsert)

        return {
            self.OUTPUT_LINES: lines_id,
            self.OUTPUT_POLYGONS: polys_id,
        }
