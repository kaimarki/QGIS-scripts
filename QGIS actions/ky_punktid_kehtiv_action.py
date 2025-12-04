import json
import urllib.request
import urllib.parse

from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsFields, QgsField, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsMessageLog, Qgis
)
from PyQt5.QtCore import QVariant

# 1) Get tunnus from clicked feature
tunnus = '[% "tunnus" %]'.strip()

if not tunnus:
    iface.messageBar().pushMessage(
        "KY punktid",
        "Tunnus puudub (veeru 'tunnus' väärtus on tühi).",
        level=Qgis.Warning,
        duration=5
    )
    raise Exception("Missing tunnus")

# 2) Build WFS request URL
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

# 3) Download JSON data
try:
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
except Exception as e:
    iface.messageBar().pushMessage(
        "KY punktid",
        f"Viga WFS päringul: {e}",
        level=Qgis.Critical,
        duration=8
    )
    raise

features_json = data.get("features", [])

if not features_json:
    iface.messageBar().pushMessage(
        "KY punktid",
        f"Punkte ei leitud (tunnus {tunnus}).",
        level=Qgis.Info,
        duration=5
    )
    raise Exception("No features returned")

# 4) Get or create target layer (memory layer)
layer_name = "ky_punktid_kehtiv"
project = QgsProject.instance()
layers = project.mapLayersByName(layer_name)

if layers:
    layer = layers[0]
else:
    # Create a new memory layer for points in EPSG:3301
    layer = QgsVectorLayer("Point?crs=EPSG:3301", layer_name, "memory")
    provider = layer.dataProvider()

    # Create fields from properties of first feature
    props = features_json[0].get("properties", {})
    fields = QgsFields()
    for key, value in props.items():
        if isinstance(value, bool):
            qtype = QVariant.Bool
        elif isinstance(value, int):
            qtype = QVariant.LongLong
        elif isinstance(value, float):
            qtype = QVariant.Double
        else:
            qtype = QVariant.String
        fields.append(QgsField(key, qtype))

    provider.addAttributes(fields)
    layer.updateFields()

    project.addMapLayer(layer)

provider = layer.dataProvider()
fields = layer.fields()

# 5) Convert WFS features to QGIS features and append
new_feats = []

for fjson in features_json:
    geom = fjson.get("geometry")
    if not geom or geom.get("type") != "Point":
        continue

    coords = geom.get("coordinates", None)
    if not coords or len(coords) < 2:
        continue

    x, y = coords[0], coords[1]
    qgis_feat = QgsFeature(fields)
    qgis_feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))

    props = fjson.get("properties", {})
    for key, value in props.items():
        if key in fields.names():
            qgis_feat[key] = value

    new_feats.append(qgis_feat)

if not new_feats:
    iface.messageBar().pushMessage(
        "KY punktid",
        f"Punkte ei õnnestunud luua (tunnus {tunnus}).",
        level=Qgis.Warning,
        duration=5
    )
    raise Exception("No valid features created")

provider.addFeatures(new_feats)
layer.updateExtents()
layer.triggerRepaint()

iface.messageBar().pushMessage(
    "KY punktid",
    f"Lisati {len(new_feats)} punkti (tunnus {tunnus}).",
    level=Qgis.Info,
    duration=4
)
