import json
import urllib.request
import urllib.parse

from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsFields, QgsField, QgsSymbol, QgsRuleBasedRenderer, Qgis
)
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor


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

# 2) WFS request
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

# 4) Define output fields
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

# 5) Create or reuse memory layer
layer_name = "valis_segments"
project = QgsProject.instance()

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
    project.addMapLayer(layer)

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

# 7) Styling: segment_type → color, width 0.5
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

iface.messageBar().pushMessage(
    "VALIS lõigud",
    f"Loodi {len(new_feats)} segmenti (tunnus {tunnus}).",
    level=Qgis.Info,
    duration=5
)
