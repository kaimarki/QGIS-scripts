from collections import defaultdict

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsFeature,
    QgsFeatureSink,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterDistance,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsVectorLayer,
)
from qgis import processing


class MergeLinesRemoveDangles(QgsProcessingAlgorithm):
    """
    Take a (multi)line layer, remove short dangling branches,
    then merge the remaining lines into long LineString(s).
    """

    INPUT = "INPUT"
    TOLERANCE = "TOLERANCE"
    MIN_DANGLE = "MIN_DANGLE"
    OUTPUT = "OUTPUT"

    def tr(self, string):
        return QCoreApplication.translate("MergeLinesRemoveDangles", string)

    def createInstance(self):
        return MergeLinesRemoveDangles()

    def name(self):
        return "mergelines_removedangles"

    def displayName(self):
        return self.tr("Merge lines & remove dangles")

    def group(self):
        return self.tr("Custom")

    def groupId(self):
        return "custom"

    def shortHelpString(self):
        return self.tr(
            "Takes a (multi)line layer, iteratively removes short dangling branches, "
            "and merges the remaining lines into long LineString features.\n\n"
            "Tolerance: snapping tolerance (layer units) when matching endpoints.\n"
            "Minimum dangling length: any line with at least one degree-1 endpoint "
            "and shorter than this length will be removed (possibly in several passes).\n"
            "All attributes are dropped; only geometry is kept in the output."
        )

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                self.tr("Input lines"),
                [QgsProcessing.TypeVectorLine],
            )
        )

        self.addParameter(
            QgsProcessingParameterDistance(
                self.TOLERANCE,
                self.tr("Endpoint snapping tolerance"),
                defaultValue=0.0,
                parentParameterName=self.INPUT,
            )
        )

        self.addParameter(
            QgsProcessingParameterDistance(
                self.MIN_DANGLE,
                self.tr("Minimum dangling length to remove"),
                defaultValue=12.0,
                parentParameterName=self.INPUT,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr("Merged lines"),
                QgsProcessing.TypeVectorLine,
            )
        )

    # ------------------------------------------------------------------ #
    # Core algorithm
    # ------------------------------------------------------------------ #
    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException(self.invalidSourceError(parameters, self.INPUT))

        tol = self.parameterAsDouble(parameters, self.TOLERANCE, context)
        min_dangle = self.parameterAsDouble(parameters, self.MIN_DANGLE, context)

        if feedback.isCanceled():
            return {}

        # Helper to snap endpoint coordinates into a node key
        def node_key(pt):
            if tol and tol > 0:
                x = round(pt.x() / tol) * tol
                y = round(pt.y() / tol) * tol
            else:
                x = pt.x()
                y = pt.y()
            return (x, y)

        # ------------------------------------------------------------------
        # 1. Read input and build a list of segments (each geometry part)
        # ------------------------------------------------------------------
        segments = []  # list of dicts: {points, length, k1, k2}
        nodes = defaultdict(list)  # node_key -> list of segment indices

        feedback.pushInfo("Reading input features and building node graph…")

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

        if not segments:
            feedback.pushInfo("No line segments found in input.")
            sink, dest_id = self.parameterAsSink(
                parameters,
                self.OUTPUT,
                context,
                source.fields(),
                source.wkbType(),
                source.sourceCrs(),
            )
            return {self.OUTPUT: dest_id}

        # ------------------------------------------------------------------
        # 2. Iteratively remove dangling segments shorter than min_dangle
        # ------------------------------------------------------------------
        feedback.pushInfo("Removing dangling segments…")

        remaining = set(range(len(segments)))
        changed = True

        while changed and not feedback.isCanceled():
            changed = False

            # Compute degrees for current remaining graph
            deg = {}
            for nk, sids in nodes.items():
                c = sum(1 for sid in sids if sid in remaining)
                if c > 0:
                    deg[nk] = c

            # Try to peel off short dangling segments
            for sid in list(remaining):
                seg = segments[sid]
                d1 = deg.get(seg["k1"], 0)
                d2 = deg.get(seg["k2"], 0)

                if seg["length"] < min_dangle and (d1 == 1 or d2 == 1):
                    remaining.remove(sid)
                    changed = True

        feedback.pushInfo(f"Remaining segments after dangle removal: {len(remaining)}")

        # ------------------------------------------------------------------
        # 3. Put cleaned segments into a temporary memory layer
        # ------------------------------------------------------------------
        if feedback.isCanceled():
            return {}

        crs = source.sourceCrs()
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

        # ------------------------------------------------------------------
        # 4. Dissolve and line-merge to get single LineString(s)
        # ------------------------------------------------------------------
        feedback.pushInfo("Dissolving and merging lines…")

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

        # ------------------------------------------------------------------
        # 5. Copy into the output sink
        # ------------------------------------------------------------------
        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            merged_layer.fields(),
            merged_layer.wkbType(),
            merged_layer.sourceCrs(),
        )

        for f in merged_layer.getFeatures():
            if feedback.isCanceled():
                break
            sink.addFeature(f, QgsFeatureSink.FastInsert)

        return {self.OUTPUT: dest_id}
