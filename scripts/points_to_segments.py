from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterFeatureSink,
    QgsProcessingException,
    QgsFeature,
    QgsFields,
    QgsField,
    QgsWkbTypes,
    QgsFeatureSink,
    QgsGeometry,
)


class ValisPointsToSegments(QgsProcessingAlgorithm):
    """
    Loob VALIS piiripunktidest järjestatud lõigud
    """

    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def initAlgorithm(self, config=None):
        # Sisend: punktikiht
        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT,
                "Sisend punktikiht",
                [QgsProcessing.TypeVectorPoint],
            )
        )

        # Väljund: joonkiht
        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                "Väljund lõigud",
                QgsProcessing.TypeVectorLine,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source = self.parameterAsSource(parameters, self.INPUT, context)
        if source is None:
            raise QgsProcessingException("Sisendkihti ei õnnestunud lugeda.")

        src_fields = source.fields()

        # Kontroll, et vajalikud väljad eksisteerivad
        required = ["piiri_tyyp", "piiripunkti_nr", "punkti_jnr", "piiripunkt"]
        missing = [f for f in required if src_fields.indexFromName(f) == -1]
        if missing:
            raise QgsProcessingException(
                "Puuduvad vajalikud väljad: {}".format(", ".join(missing))
            )

        idx_piiri_tyyp = src_fields.indexFromName("piiri_tyyp")
        idx_pp_nr = src_fields.indexFromName("piiripunkti_nr")
        idx_p_jnr = src_fields.indexFromName("punkti_jnr")
        idx_piiripunkt = src_fields.indexFromName("piiripunkt")

        # Väljundväljad – kasutame algsete väljade tüüpe
        out_fields = QgsFields()

        f_pp_nr = src_fields.field(idx_pp_nr)
        f_p_jnr = src_fields.field(idx_p_jnr)
        f_piiripunkt = src_fields.field(idx_piiripunkt)

        out_fields.append(
            QgsField(
                "piiripunkti_nr_from",
                f_pp_nr.type(),
                f_pp_nr.typeName(),
                f_pp_nr.length(),
                f_pp_nr.precision(),
            )
        )
        out_fields.append(
            QgsField(
                "piiripunkti_nr_to",
                f_pp_nr.type(),
                f_pp_nr.typeName(),
                f_pp_nr.length(),
                f_pp_nr.precision(),
            )
        )

        out_fields.append(
            QgsField(
                "punkti_jnr_from",
                f_p_jnr.type(),
                f_p_jnr.typeName(),
                f_p_jnr.length(),
                f_p_jnr.precision(),
            )
        )
        out_fields.append(
            QgsField(
                "punkti_jnr_to",
                f_p_jnr.type(),
                f_p_jnr.typeName(),
                f_p_jnr.length(),
                f_p_jnr.precision(),
            )
        )

        out_fields.append(
            QgsField(
                "piiripunkt_from",
                f_piiripunkt.type(),
                f_piiripunkt.typeName(),
                f_piiripunkt.length(),
                f_piiripunkt.precision(),
            )
        )
        out_fields.append(
            QgsField(
                "piiripunkt_to",
                f_piiripunkt.type(),
                f_piiripunkt.typeName(),
                f_piiripunkt.length(),
                f_piiripunkt.precision(),
            )
        )

        # segment_type – lihtne tekstiväli
        out_fields.append(QgsField("segment_type", QVariant.String, "string", 20))

        sink, dest_id = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            QgsWkbTypes.LineString,
            source.sourceCrs(),
        )

        if sink is None:
            raise QgsProcessingException("Väljundkihti ei õnnestunud luua.")

        # 1) Kogume ainult piiri_tyyp = 'VALIS' punktid
        points = []
        for feat in source.getFeatures():
            if feedback.isCanceled():
                break

            attrs = feat.attributes()
            piiri_tyyp_val = attrs[idx_piiri_tyyp]

            if piiri_tyyp_val != "VALIS":
                continue

            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue

            try:
                pt = geom.asPoint()
            except Exception:
                if geom.isMultipart():
                    pts = geom.asMultiPoint()
                    if not pts:
                        continue
                    pt = pts[0]
                else:
                    continue

            jnr_raw = attrs[idx_p_jnr]
            try:
                jnr = int(jnr_raw)
            except Exception:
                jnr = jnr_raw

            points.append((jnr, feat, pt))

        # 2) Sorteerime punktid punkti_jnr järgi
        points.sort(key=lambda x: x[0])

        if len(points) < 2:
            # Pole midagi ühendamiseks
            return {self.OUTPUT: dest_id}

        total = len(points)
        for i, (jnr_from, feat_from, pt_from) in enumerate(points):
            if feedback.isCanceled():
                break

            # Järgmine punkt, viimane ühendatakse esimese punktiga
            jnr_to, feat_to, pt_to = points[(i + 1) % total]

            attrs_from = feat_from.attributes()
            attrs_to = feat_to.attributes()

            piiripunkti_nr_from = attrs_from[idx_pp_nr]
            piiripunkti_nr_to = attrs_to[idx_pp_nr]

            punkti_jnr_from = attrs_from[idx_p_jnr]
            punkti_jnr_to = attrs_to[idx_p_jnr]

            piiripunkt_from = attrs_from[idx_piiripunkt]
            piiripunkt_to = attrs_to[idx_piiripunkt]

            # segment_type loogika
            segment_type = self._segment_type(piiripunkt_from, piiripunkt_to)

            out_feat = QgsFeature(out_fields)
            # *** OLULINE MUUTUS: kasutame fromPolylineXY, mis ootab QgsPointXY ***
            out_geom = QgsGeometry.fromPolylineXY([pt_from, pt_to])
            out_feat.setGeometry(out_geom)

            out_feat.setAttributes(
                [
                    piiripunkti_nr_from,
                    piiripunkti_nr_to,
                    punkti_jnr_from,
                    punkti_jnr_to,
                    piiripunkt_from,
                    piiripunkt_to,
                    segment_type,
                ]
            )

            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            if total > 0:
                feedback.setProgress(int(100.0 * (i + 1) / total))

        return {self.OUTPUT: dest_id}

    @staticmethod
    def _segment_type(piiripunkt_from, piiripunkt_to):
        """
        Vastab kirjeldatud CASE-reeglile.
        """
        water_val = "VEEKOGU_TELJE_PUNKT"

        from_is_water = piiripunkt_from == water_val
        to_is_water = piiripunkt_to == water_val

        if from_is_water and to_is_water:
            return "water"
        if from_is_water != to_is_water:
            return "interpolate"
        return "land"

    def createInstance(self):
        return ValisPointsToSegments()

    def name(self):
        return "valis_points_to_segments"

    def displayName(self):
        return "VALIS piiripunktidest lõigud"

    def group(self):
        return "Custom"

    def groupId(self):
        return "custom"

    def shortHelpString(self):
        return (
            "Loob sisendpunktide põhjal joonkihi, kus iga lõik on kahe järjestikuse "
            "piiripunkti (piiri_tyyp = 'VALIS') vaheline segment. Lõigule lisatakse "
            "mõlema otspunkti atribuudid ja segment_type väli ('water', 'interpolate', 'land')."
        )
