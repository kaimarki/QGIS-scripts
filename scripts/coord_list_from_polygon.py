from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterVectorLayer,
    QgsField,
    QgsWkbTypes,
    edit
)

class CoordListFromPolygon(QgsProcessingAlgorithm):
    """
    Creates a coordinate list from polygon geometries
    and stores it in a field 'bananamaster69000'.
    """

    INPUT = 'INPUT'
    FIELD_NAME = 'QGIS_numba_uno'

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.INPUT,
                'Input layer (polygons)',
                [QgsProcessing.TypeVectorPolygon]
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        layer = self.parameterAsVectorLayer(parameters, self.INPUT, context)
        if layer is None:
            raise QgsProcessingException('No valid input layer.')

        provider = layer.dataProvider()

        # Create field if it doesn't exist yet
        field_names = [f.name() for f in provider.fields()]
        if self.FIELD_NAME not in field_names:
            provider.addAttributes([QgsField(self.FIELD_NAME, QVariant.String)])
            layer.updateFields()

        idx = layer.fields().indexOf(self.FIELD_NAME)

        total = layer.featureCount()
        if total == 0:
            return {}

        with edit(layer):
            for i, feat in enumerate(layer.getFeatures()):
                if feedback.isCanceled():
                    break

                geom = feat.geometry()
                coord_text = None

                if geom and not geom.isEmpty() and QgsWkbTypes.geometryType(geom.wkbType()) == QgsWkbTypes.PolygonGeometry:
                    # Collect points from outer rings
                    points = []

                    if QgsWkbTypes.isMultiType(geom.wkbType()):
                        # MultiPolygon: list of polygons, each is [outer_ring, inner_ring1, ...]
                        for poly in geom.asMultiPolygon():
                            if not poly:
                                continue
                            outer_ring = poly[0]
                            points.extend(outer_ring)
                    else:
                        # Single polygon
                        poly = geom.asPolygon()
                        if poly:
                            outer_ring = poly[0]
                            points.extend(outer_ring)

                    # Build coordinate list text: [x, y], [x, y], ...
                    if points:
                        coord_pairs = []
                        for pt in points:
                            coord_pairs.append(f'[{pt.x()}, {pt.y()}]')
                        coord_text = ', '.join(coord_pairs)

                feat[self.FIELD_NAME] = coord_text
                layer.updateFeature(feat)

                if total > 0:
                    feedback.setProgress(int(100 * (i + 1) / total))

        return {}

    def name(self):
        return 'coord_list_from_polygon'

    def displayName(self):
        return 'Create coordinate list from polygon'

    def group(self):
        return 'Custom'

    def groupId(self):
        return 'custom'

    def createInstance(self):
        return CoordListFromPolygon()
