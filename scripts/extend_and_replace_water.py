import processing
from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFeatureSink,
    QgsFeature,
    QgsGeometry,
    QgsWkbTypes,
    QgsFields,
    QgsField,
    QgsPointXY,
    QgsFeatureSink,
    QgsSpatialIndex,
)

class SigmaExtendAndReplaceWaterStep2(QgsProcessingAlgorithm):
    IN_SEGMENTS   = "IN_SEGMENTS"
    IN_REF_POLY   = "IN_REF_POLY"
    AOI_BUFFER_M  = "AOI_BUFFER_M"
    EXTRA_M       = "EXTRA_M"
    SNAP_TOL      = "SNAP_TOL"
    OUT_SEGMENTS  = "OUT_SEGMENTS"
    OUT_POLYGONS  = "OUT_POLYGONS"   # NEW

    def createInstance(self):
        return SigmaExtendAndReplaceWaterStep2()

    def name(self):
        return "extend_and_replace_water"

    def displayName(self):
        return " Alpha - extend interpolate + replace water by shoreline"

    def group(self):
        return "Custom"

    def groupId(self):
        return "custom"

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.IN_SEGMENTS,
                "Cadastre segment layer (valis_segments)",
                [QgsProcessing.TypeVectorLine]
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.IN_REF_POLY,
                "Reference polygon layer (polygons)",
                [QgsProcessing.TypeVectorPolygon]
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AOI_BUFFER_M,
                "AOI buffer around unit polygon (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=100.0,
                minValue=0.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.EXTRA_M,
                "Extend AFTER intersection (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.0,
                minValue=0.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SNAP_TOL,
                "Snap tolerance to shoreline ring (m)",
                type=QgsProcessingParameterNumber.Double,
                defaultValue=0.5,
                minValue=0.0
            )
        )
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_SEGMENTS,
                "Segments (interpolate extended, water replaced)",
                QgsProcessing.TypeVectorLine
            )
        )

        # NEW: single polygon result (one feature)
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUT_POLYGONS,
                "Result polygon (single ring from final segments)",
                QgsProcessing.TypeVectorPolygon
            )
        )

    # -------------------------
    # Basic helpers
    # -------------------------
    def _require_field(self, source, name):
        idx = source.fields().indexFromName(name)
        if idx < 0:
            raise QgsProcessingException(f"Missing required field: {name}")
        return idx

    def _line_endpoints(self, geom):
        if geom is None or geom.isEmpty():
            raise QgsProcessingException("Segment geometry is empty.")
        if geom.isMultipart():
            parts = geom.asMultiPolyline()
            if not parts or not parts[0] or len(parts[0]) < 2:
                raise QgsProcessingException("Multipart line has insufficient vertices.")
            pts = parts[0]
        else:
            pts = geom.asPolyline()
            if not pts or len(pts) < 2:
                raise QgsProcessingException("Line has insufficient vertices.")
        return QgsPointXY(pts[0]), QgsPointXY(pts[-1])

    def _as_polyline_points(self, geom):
        if geom is None or geom.isEmpty():
            return []
        if geom.isMultipart():
            m = geom.asMultiPolyline()
            return m[0] if m else []
        return geom.asPolyline()

    def _build_ordered_ring(self, features, idx_jfrom, idx_jto, idx_segtype):
        segs = {}
        for f in features:
            j_from = f.attributes()[idx_jfrom]
            j_to   = f.attributes()[idx_jto]
            if j_from is None or j_to is None:
                raise QgsProcessingException("punkti_jnr_from/to contains NULL.")
            try:
                j_from = int(j_from)
                j_to   = int(j_to)
            except Exception:
                raise QgsProcessingException("punkti_jnr_from/to must be integers (or castable).")
            if j_from in segs:
                raise QgsProcessingException(f"Duplicate punkti_jnr_from: {j_from}")

            p0, p1 = self._line_endpoints(f.geometry())
            st = (f.attributes()[idx_segtype] or "").strip().lower()
            if st not in ("land", "water", "interpolate"):
                raise QgsProcessingException(f"Invalid segment_type '{st}' for punkti_jnr_from={j_from}")

            segs[j_from] = {
                "feat": f,
                "j_from": j_from,
                "j_to": j_to,
                "p0": p0,
                "p1": p1,
                "type": st
            }

        if not segs:
            raise QgsProcessingException("No segments in input.")

        start = min(segs.keys())
        ordered = []
        cur = start
        guard = 0
        while True:
            guard += 1
            if guard > len(segs) + 2:
                raise QgsProcessingException("Segment loop did not close (bad ordering).")
            if cur not in segs:
                raise QgsProcessingException(f"Missing segment for punkti_jnr_from={cur}")
            ordered.append(segs[cur])
            cur = segs[cur]["j_to"]
            if cur == start:
                break

        if len(ordered) != len(segs):
            raise QgsProcessingException("Ordering did not include all segments (broken ring).")

        return ordered

    def _polygon_from_ordered(self, ordered):
        coords = [ordered[0]["p0"]]
        for s in ordered:
            last = coords[-1]
            p0, p1 = s["p0"], s["p1"]
            if last == p0:
                coords.append(p1)
            elif last == p1:
                coords.append(p0)
            else:
                coords.append(p1)
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        poly = QgsGeometry.fromPolygonXY([coords])
        if poly.isEmpty():
            raise QgsProcessingException("Polygon creation failed (empty).")
        if not poly.isGeosValid():
            poly2 = poly.makeValid()
            if poly2 is None or poly2.isEmpty():
                raise QgsProcessingException("Polygon invalid and makeValid() returned empty.")
            poly = poly2
        return poly

    def _unit_dir(self, a: QgsPointXY, b: QgsPointXY):
        dx = b.x() - a.x()
        dy = b.y() - a.y()
        length = (dx*dx + dy*dy) ** 0.5
        if length == 0:
            raise QgsProcessingException("Zero-length direction for interpolate segment.")
        return dx/length, dy/length

    def _extract_points(self, geom: QgsGeometry):
        if geom is None or geom.isEmpty():
            return []

        w = geom.wkbType()
        gtype = QgsWkbTypes.geometryType(w)

        if gtype == QgsWkbTypes.PointGeometry:
            if QgsWkbTypes.isMultiType(w):
                return [QgsPointXY(p) for p in geom.asMultiPoint()]
            return [QgsPointXY(geom.asPoint())]

        pts = []
        if gtype == QgsWkbTypes.LineGeometry:
            if QgsWkbTypes.isMultiType(w):
                for pl in geom.asMultiPolyline():
                    for p in pl:
                        pts.append(QgsPointXY(p))
            else:
                for p in geom.asPolyline():
                    pts.append(QgsPointXY(p))
            return pts

        if gtype == QgsWkbTypes.PolygonGeometry:
            if QgsWkbTypes.isMultiType(w):
                for poly in geom.asMultiPolygon():
                    if poly and poly[0]:
                        for p in poly[0]:
                            pts.append(QgsPointXY(p))
            else:
                poly = geom.asPolygon()
                if poly and poly[0]:
                    for p in poly[0]:
                        pts.append(QgsPointXY(p))
            return pts

        try:
            for p in geom.vertices():
                pts.append(QgsPointXY(p))
        except Exception:
            pass
        return pts

    def _dissolve_geoms(self, geoms):
        if not geoms:
            return None
        try:
            u = QgsGeometry.unaryUnion(geoms)
            if u is not None and not u.isEmpty():
                if not u.isGeosValid():
                    u2 = u.makeValid()
                    if u2 is not None and not u2.isEmpty():
                        u = u2
                return u
        except Exception:
            pass

        g = QgsGeometry(geoms[0])
        for gg in geoms[1:]:
            g = g.combine(gg)
        if g is None or g.isEmpty():
            return None
        if not g.isGeosValid():
            g2 = g.makeValid()
            if g2 is not None and not g2.isEmpty():
                g = g2
        return g

    def _ray_first_hit_on_union(self, origin: QgsPointXY, dirx, diry, union_poly: QgsGeometry):
        L = 20000.0
        end = QgsPointXY(origin.x() + dirx * L, origin.y() + diry * L)
        ray = QgsGeometry.fromPolylineXY([origin, end])

        inter = ray.intersection(union_poly)
        pts = self._extract_points(inter)
        if not pts:
            raise QgsProcessingException("Ray did not intersect dissolved reference polygons.")

        best_t = None
        best_pt = None
        for p in pts:
            vx = p.x() - origin.x()
            vy = p.y() - origin.y()
            t = vx * dirx + vy * diry
            if t <= 0:
                continue
            if best_t is None or t < best_t:
                best_t = t
                best_pt = p

        if best_pt is None:
            raise QgsProcessingException("Ray intersects, but no forward intersection point found.")

        return best_pt

    def _extract_exterior_ring_lines(self, poly: QgsGeometry):
        if poly is None or poly.isEmpty():
            return []
        if not poly.isGeosValid():
            poly2 = poly.makeValid()
            if poly2 is not None and not poly2.isEmpty():
                poly = poly2

        w = poly.wkbType()
        if QgsWkbTypes.geometryType(w) != QgsWkbTypes.PolygonGeometry:
            return []

        lines = []
        if QgsWkbTypes.isMultiType(w):
            mp = poly.asMultiPolygon()
            for rings in mp:
                if not rings or not rings[0] or len(rings[0]) < 4:
                    continue
                lines.append(QgsGeometry.fromPolylineXY([QgsPointXY(p) for p in rings[0]]))
        else:
            rings = poly.asPolygon()
            if rings and rings[0] and len(rings[0]) >= 4:
                lines.append(QgsGeometry.fromPolylineXY([QgsPointXY(p) for p in rings[0]]))
        return lines

    def _nearest_point_on_line(self, line: QgsGeometry, pt: QgsPointXY):
        pgeom = QgsGeometry.fromPointXY(pt)
        np = line.nearestPoint(pgeom)
        if np is None or np.isEmpty():
            return None, None
        npt = QgsPointXY(np.asPoint())
        dist = np.distance(pgeom)
        return npt, dist

    def _split_line_by_point(self, line: QgsGeometry, pt: QgsPointXY):
        g = QgsGeometry(line)
        pts = [pt]
        try:
            res = g.splitGeometry(pts, False)
        except Exception as e:
            raise QgsProcessingException(f"splitGeometry failed: {e}")

        if not isinstance(res, (tuple, list)) or len(res) < 2:
            raise QgsProcessingException("splitGeometry returned unexpected result.")
        new_geoms = res[1]

        parts = []
        if g is not None and not g.isEmpty():
            parts.append(g)
        if new_geoms:
            for ng in new_geoms:
                if ng and not ng.isEmpty():
                    parts.append(ng)

        out = []
        for p in parts:
            if QgsWkbTypes.geometryType(p.wkbType()) == QgsWkbTypes.LineGeometry:
                out.append(p)
        return out

    def _choose_shore_path_by_splitting(self, ring_line: QgsGeometry, p1: QgsPointXY, p2: QgsPointXY, water_geom: QgsGeometry, tol: float):
        parts1 = self._split_line_by_point(ring_line, p1)
        parts2 = []
        for part in parts1:
            parts2.extend(self._split_line_by_point(part, p2))

        candidates = []
        for g in parts2:
            pts = self._as_polyline_points(g)
            if not pts or len(pts) < 2:
                continue
            a = QgsPointXY(pts[0])
            b = QgsPointXY(pts[-1])

            d_a_p1 = QgsGeometry.fromPointXY(a).distance(QgsGeometry.fromPointXY(p1))
            d_b_p1 = QgsGeometry.fromPointXY(b).distance(QgsGeometry.fromPointXY(p1))
            d_a_p2 = QgsGeometry.fromPointXY(a).distance(QgsGeometry.fromPointXY(p2))
            d_b_p2 = QgsGeometry.fromPointXY(b).distance(QgsGeometry.fromPointXY(p2))

            ok = ((d_a_p1 <= tol and d_b_p2 <= tol) or (d_b_p1 <= tol and d_a_p2 <= tol))
            if ok:
                candidates.append(g)

        if len(candidates) < 2:
            if not candidates:
                best = None
                best_d = None
                for g in parts2:
                    d = g.distance(water_geom)
                    if best_d is None or d < best_d:
                        best_d = d
                        best = g
                if best is None:
                    raise QgsProcessingException("Could not derive shoreline path candidates after splitting.")
                return best
            return candidates[0]

        best = None
        best_d = None
        for g in candidates:
            d = g.distance(water_geom)
            if best_d is None or d < best_d:
                best_d = d
                best = g
        return best

    # -------------------------
    # NEW: split a polyline into N consecutive parts by length
    # -------------------------
    def _split_polyline_into_n_parts(self, line: QgsGeometry, n: int):
        if n <= 0:
            raise QgsProcessingException("n must be > 0")
        pts = self._as_polyline_points(line)
        if not pts or len(pts) < 2:
            raise QgsProcessingException("Cannot split: line has <2 vertices")

        def seg_len(a, b):
            dx = b.x() - a.x()
            dy = b.y() - a.y()
            return (dx*dx + dy*dy) ** 0.5

        total = 0.0
        for i in range(len(pts) - 1):
            total += seg_len(pts[i], pts[i+1])
        if total == 0:
            raise QgsProcessingException("Cannot split: zero-length line")

        targets = [total * (i / n) for i in range(1, n)]
        parts_pts = []
        cur_part = [QgsPointXY(pts[0])]

        walked = 0.0
        t_idx = 0

        for i in range(len(pts) - 1):
            a = QgsPointXY(pts[i])
            b = QgsPointXY(pts[i+1])
            L = seg_len(a, b)
            if L == 0:
                continue

            while t_idx < len(targets) and walked + L >= targets[t_idx]:
                need = targets[t_idx] - walked
                r = need / L
                cut = QgsPointXY(a.x() + (b.x() - a.x()) * r, a.y() + (b.y() - a.y()) * r)
                cur_part.append(cut)
                parts_pts.append(cur_part)
                cur_part = [cut]
                t_idx += 1

            cur_part.append(b)
            walked += L

        parts_pts.append(cur_part)

        if len(parts_pts) != n:
            if len(parts_pts) < n:
                raise QgsProcessingException(f"Split produced too few parts: {len(parts_pts)} != {n}")
            merged = parts_pts[:n-1]
            last = []
            for p in parts_pts[n-1:]:
                if not last:
                    last = p
                else:
                    last.extend(p[1:])
            merged.append(last)
            parts_pts = merged

        out = []
        for p in parts_pts:
            if len(p) < 2:
                raise QgsProcessingException("Split produced degenerate part")
            out.append(QgsGeometry.fromPolylineXY(p))
        return out

    # -------------------------
    # NEW: stitch final lines -> single polygon
    # -------------------------
    def _same_pt(self, a: QgsPointXY, b: QgsPointXY, tol=1e-8):
        dx = a.x() - b.x()
        dy = a.y() - b.y()
        return (dx*dx + dy*dy) <= tol*tol

    def _geom_polyline_pts(self, g: QgsGeometry):
        pts = self._as_polyline_points(g)
        return [QgsPointXY(p) for p in pts] if pts else []

    def _append_oriented(self, ring_pts, part_pts):
        if not part_pts:
            return ring_pts
        if not ring_pts:
            return part_pts[:]  # start ring

        last = ring_pts[-1]

        d0 = QgsGeometry.fromPointXY(last).distance(QgsGeometry.fromPointXY(part_pts[0]))
        d1 = QgsGeometry.fromPointXY(last).distance(QgsGeometry.fromPointXY(part_pts[-1]))
        if d1 < d0:
            part_pts = list(reversed(part_pts))

        # join without duplicating the junction vertex
        if self._same_pt(part_pts[0], ring_pts[-1]):
            ring_pts.extend(part_pts[1:])
        else:
            ring_pts.extend(part_pts)

        # remove any accidental consecutive duplicates
        cleaned = [ring_pts[0]]
        for p in ring_pts[1:]:
            if not self._same_pt(p, cleaned[-1]):
                cleaned.append(p)
        return cleaned

    # -------------------------
    # Main
    # -------------------------
    def processAlgorithm(self, parameters, context, feedback):
        seg_source = self.parameterAsSource(parameters, self.IN_SEGMENTS, context)
        ref_source = self.parameterAsSource(parameters, self.IN_REF_POLY, context)

        extra_m  = float(self.parameterAsDouble(parameters, self.EXTRA_M, context))
        aoi_m    = float(self.parameterAsDouble(parameters, self.AOI_BUFFER_M, context))
        snap_tol = float(self.parameterAsDouble(parameters, self.SNAP_TOL, context))

        if seg_source is None:
            raise QgsProcessingException("Invalid segment input.")
        if ref_source is None:
            raise QgsProcessingException("Invalid reference polygon input.")

        idx_jfrom = self._require_field(seg_source, "punkti_jnr_from")
        idx_jto   = self._require_field(seg_source, "punkti_jnr_to")
        idx_st    = self._require_field(seg_source, "segment_type")

        feats = list(seg_source.getFeatures())
        if not feats:
            raise QgsProcessingException("No segments provided.")

        ordered = self._build_ordered_ring(feats, idx_jfrom, idx_jto, idx_st)

        interp_idx = [i for i, s in enumerate(ordered) if s["type"] == "interpolate"]
        if len(interp_idx) != 2:
            raise QgsProcessingException(f"Expected exactly 2 interpolate segments for this step, got {len(interp_idx)}.")

        for i in interp_idx:
            prev_t = ordered[(i - 1) % len(ordered)]["type"]
            next_t = ordered[(i + 1) % len(ordered)]["type"]
            if prev_t == "interpolate" or next_t == "interpolate":
                raise QgsProcessingException("ERROR: interpolate touches interpolate (data error).")
            if not ((prev_t == "land" and next_t == "water") or (prev_t == "water" and next_t == "land")):
                raise QgsProcessingException(f"ERROR: interpolate must be between land and water. Got prev={prev_t}, next={next_t}.")

        unit_poly = self._polygon_from_ordered(ordered)
        if aoi_m > 0:
            aoi = unit_poly.buffer(aoi_m, 8)
            if aoi is None or aoi.isEmpty():
                raise QgsProcessingException("AOI buffer returned empty geometry.")
        else:
            aoi = unit_poly

        ref_index = QgsSpatialIndex()
        ref_geom_cache = {}
        for rf in ref_source.getFeatures():
            ref_index.addFeature(rf)
            ref_geom_cache[rf.id()] = rf.geometry()

        cand_ids = ref_index.intersects(aoi.boundingBox())
        if not cand_ids:
            raise QgsProcessingException("No reference polygons found in AOI bbox.")

        selected = []
        for fid in cand_ids:
            g = ref_geom_cache.get(fid)
            if g is None or g.isEmpty():
                continue
            if g.intersects(aoi):
                selected.append(g)

        if not selected:
            raise QgsProcessingException("No reference polygons intersect AOI (after bbox prefilter).")

        union_poly = self._dissolve_geoms(selected)
        if union_poly is None or union_poly.isEmpty():
            raise QgsProcessingException("Dissolve/union of reference polygons failed (empty).")

        interp_hit = {}
        interp_ext_geom = {}

        for i in interp_idx:
            s = ordered[i]
            f_in = s["feat"]

            prev_t = ordered[(i - 1) % len(ordered)]["type"]
            next_t = ordered[(i + 1) % len(ordered)]["type"]

            p0, p1 = self._line_endpoints(f_in.geometry())

            if prev_t == "land" and next_t == "water":
                fixed = p0
                movable = p1
                fixed_end = "from"
            else:
                fixed = p1
                movable = p0
                fixed_end = "to"

            dirx, diry = self._unit_dir(fixed, movable)
            hit_pt = self._ray_first_hit_on_union(fixed, dirx, diry, union_poly)

            ext_pt = QgsPointXY(hit_pt.x() + dirx * extra_m, hit_pt.y() + diry * extra_m)
            ext_geom = QgsGeometry.fromPolylineXY([fixed, ext_pt])

            interp_hit[i] = {"hit": hit_pt, "fixed_end": fixed_end}
            interp_ext_geom[i] = ext_geom

        i1, i2 = sorted(interp_idx)
        between = list(range(i1 + 1, i2))
        between_wrap = list(range(i2 + 1, len(ordered))) + list(range(0, i1))

        def is_all_water(idxs):
            return len(idxs) > 0 and all(ordered[j]["type"] == "water" for j in idxs)

        if is_all_water(between) and not is_all_water(between_wrap):
            water_idxs = between
            interp_a = i1
            interp_b = i2
        elif is_all_water(between_wrap) and not is_all_water(between):
            water_idxs = between_wrap
            interp_a = i2
            interp_b = i1
        else:
            raise QgsProcessingException("Could not uniquely identify the water block between the two interpolate segments.")

        water_lines = []
        for j in water_idxs:
            g = ordered[j]["feat"].geometry()
            if g and not g.isEmpty():
                water_lines.append(g)
        if not water_lines:
            raise QgsProcessingException("Water block has no geometries.")
        try:
            water_geom = QgsGeometry.unaryUnion(water_lines)
        except Exception:
            water_geom = water_lines[0]
            for gg in water_lines[1:]:
                water_geom = water_geom.combine(gg)

        ring_lines = self._extract_exterior_ring_lines(union_poly)
        if not ring_lines:
            raise QgsProcessingException("Failed to extract exterior rings from dissolved polygons.")

        pA = interp_hit[interp_a]["hit"]
        pB = interp_hit[interp_b]["hit"]

        best_ring = None
        best_score = None
        best_pA_s = None
        best_pB_s = None

        for ring in ring_lines:
            pA_s, dA = self._nearest_point_on_line(ring, pA)
            pB_s, dB = self._nearest_point_on_line(ring, pB)
            if pA_s is None or pB_s is None:
                continue
            if dA is None or dB is None:
                continue
            if dA > snap_tol or dB > snap_tol:
                continue
            score = dA + dB
            if best_score is None or score < best_score:
                best_score = score
                best_ring = ring
                best_pA_s = pA_s
                best_pB_s = pB_s

        if best_ring is None:
            raise QgsProcessingException("Could not find a shoreline ring that contains both hit points (increase SNAP_TOL?).")

        shoreline_path = self._choose_shore_path_by_splitting(best_ring, best_pA_s, best_pB_s, water_geom, snap_tol * 2.0)
        if shoreline_path is None or shoreline_path.isEmpty():
            raise QgsProcessingException("Failed to build shoreline replacement path.")

        # ---------------------------------------------------------
        # Build a full replacement water line and split into k parts
        # ---------------------------------------------------------
        def _endpt_of_line(g: QgsGeometry):
            pts = self._as_polyline_points(g)
            if not pts or len(pts) < 2:
                raise QgsProcessingException("Bad interpolate geometry (no endpoints)")
            return QgsPointXY(pts[-1])

        extA = _endpt_of_line(interp_ext_geom[interp_a])
        extB = _endpt_of_line(interp_ext_geom[interp_b])

        sp = self._as_polyline_points(shoreline_path)
        if not sp or len(sp) < 2:
            raise QgsProcessingException("shoreline_path has no vertices")

        # Ensure arc is oriented from best_pA_s -> best_pB_s
        d0 = QgsGeometry.fromPointXY(QgsPointXY(sp[0])).distance(QgsGeometry.fromPointXY(best_pA_s))
        d1 = QgsGeometry.fromPointXY(QgsPointXY(sp[-1])).distance(QgsGeometry.fromPointXY(best_pA_s))
        if d1 < d0:
            sp = list(reversed(sp))

        full_pts = [extA]
        if full_pts[-1] != best_pA_s:
            full_pts.append(best_pA_s)

        for p in sp:
            q = QgsPointXY(p)
            if q != full_pts[-1]:
                full_pts.append(q)

        if best_pB_s != full_pts[-1]:
            full_pts.append(best_pB_s)
        if extB != full_pts[-1]:
            full_pts.append(extB)

        full_repl = QgsGeometry.fromPolylineXY(full_pts)
        if full_repl is None or full_repl.isEmpty():
            raise QgsProcessingException("Failed to build full replacement water line.")

        k = len(water_idxs)
        water_pieces = self._split_polyline_into_n_parts(full_repl, k)
        water_repl_by_idx = {water_idxs[t]: water_pieces[t] for t in range(k)}

        # Output segments sink
        out_fields = QgsFields(seg_source.fields())
        out_fields.append(QgsField("_moved", QVariant.Int))
        out_fields.append(QgsField("_note", QVariant.String))

        sink, sink_id = self.parameterAsSink(
            parameters,
            self.OUT_SEGMENTS,
            context,
            out_fields,
            QgsWkbTypes.LineString,
            seg_source.sourceCrs()
        )
        if sink is None:
            raise QgsProcessingException("Failed to create output sink.")

        st_idx = seg_source.fields().indexFromName("segment_type")

        # Keep final geometries for polygon build
        final_geom_by_idx = {}

        for i, s in enumerate(ordered):
            st = s["type"]
            f_in = s["feat"]

            f_out = QgsFeature(out_fields)
            attrs = list(f_in.attributes())

            if st == "interpolate":
                g_final = interp_ext_geom[i]
                f_out.setGeometry(g_final)
                f_out.setAttributes(attrs + [1, "interpolate_extended"])

            elif st == "water":
                g_final = water_repl_by_idx.get(i)
                if g_final is None or g_final.isEmpty():
                    raise QgsProcessingException("Internal error: missing replacement water piece for a water segment.")
                f_out.setGeometry(g_final)

                if st_idx >= 0:
                    attrs[st_idx] = "water"
                f_out.setAttributes(attrs + [1, "water_replaced_by_shoreline"])

            else:  # land
                g_final = f_in.geometry()
                f_out.setGeometry(g_final)
                f_out.setAttributes(attrs + [0, "land_unchanged"])

            final_geom_by_idx[i] = g_final
            sink.addFeature(f_out, QgsFeatureSink.FastInsert)

        # -------------------------
        # NEW: Build ONE polygon by stitching final line geometries in order
        # -------------------------
        ring_pts = []
        for i in range(len(ordered)):
            g = final_geom_by_idx.get(i)
            if g is None or g.isEmpty():
                raise QgsProcessingException("Missing final geometry while building polygon ring.")

            pts = self._geom_polyline_pts(g)
            if len(pts) < 2:
                raise QgsProcessingException("A segment has <2 vertices; cannot build ring.")
            ring_pts = self._append_oriented(ring_pts, pts)

        if not ring_pts or len(ring_pts) < 4:
            raise QgsProcessingException("Ring build failed (too few vertices).")

        if not self._same_pt(ring_pts[0], ring_pts[-1], tol=1e-8):
            ring_pts.append(ring_pts[0])

        poly = QgsGeometry.fromPolygonXY([ring_pts])
        if poly is None or poly.isEmpty():
            raise QgsProcessingException("Polygon creation failed (empty).")
        if not poly.isGeosValid():
            poly2 = poly.makeValid()
            if poly2 is None or poly2.isEmpty():
                raise QgsProcessingException("Polygon invalid and makeValid() returned empty.")
            poly = poly2

        # Output polygon sink (single feature)
        poly_fields = QgsFields()
        poly_fields.append(QgsField("_note", QVariant.String))

        poly_sink, poly_sink_id = self.parameterAsSink(
            parameters,
            self.OUT_POLYGONS,
            context,
            poly_fields,
            QgsWkbTypes.Polygon,
            seg_source.sourceCrs()
        )
        if poly_sink is None:
            raise QgsProcessingException("Failed to create polygon output sink.")

        pf = QgsFeature(poly_fields)
        pf.setGeometry(poly)
        pf.setAttributes(["stitched_from_final_segments"])
        poly_sink.addFeature(pf, QgsFeatureSink.FastInsert)

        return {
            self.OUT_SEGMENTS: sink_id,
            self.OUT_POLYGONS: poly_sink_id
        }
