# opcuainterface/siome5_exporter.py

from datetime import datetime, timezone
from lxml import etree


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"

NS = {
    "ua": UA_NS,
}


def generate_siome5_xml_from_db_nodes(nodes, template_path: str) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(template_path, parser)
    root = tree.getroot()

    # Update LastModified dynamically
    root.set("LastModified", _current_utc_timestamp())

    db_nodes_by_label = _build_db_node_lookup(nodes)

    for variable in root.xpath(".//ua:UAVariable", namespaces=NS):
        template_browse_name = variable.get("BrowseName", "")
        template_field_name = _clean_browse_name(template_browse_name)

        db_node = db_nodes_by_label.get(template_field_name)

        if db_node is not None:
            value = _get_latest_node_value(db_node)
            if value is None:
                value = 0
        else:
            value = 0

        # Strict mode: update only existing <Value>; do not add new tags
        _update_existing_value_only(variable, value)

    xml_body = etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=False,
    ).decode("utf-8")

    # Force double quotes in XML declaration
    final_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body

    return final_xml.encode("utf-8")


def _current_utc_timestamp() -> str:
    """
    Example:
        2026-05-19T07:30:15.123Z
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(now.microsecond / 1000):03d}Z"


def _build_db_node_lookup(nodes) -> dict:
    lookup = {}

    for node in nodes:
        label = getattr(node, "label", None)

        if not label:
            label = getattr(node, "node_id", "")

        field_name = _clean_browse_name(str(label))

        if field_name:
            lookup[field_name] = node

    return lookup


def _clean_browse_name(name: str) -> str:
    if not name:
        return ""

    name = str(name).strip()

    if ":" in name:
        name = name.split(":", 1)[-1]

    if "." in name:
        name = name.split(".")[-1]

    return name.strip()


def _get_latest_node_value(db_node):
    try:
        latest_reading = db_node.readings.first()

        if latest_reading is None:
            return None

        if hasattr(latest_reading, "typed_value"):
            return latest_reading.typed_value

        if hasattr(latest_reading, "value"):
            return latest_reading.value

        return None

    except Exception:
        return None


def _update_existing_value_only(variable, value):
    """
    Important:
    - Do not create new <Value> tags
    - Do not change SIOME5 structure
    - Only update value where <Value> already exists
    """

    value_elements = variable.xpath("./ua:Value", namespaces=NS)

    if not value_elements:
        return

    value_element = value_elements[0]
    leaf = _find_first_leaf(value_element)

    if leaf is not None:
        leaf.text = _format_value(value)


def _find_first_leaf(element):
    for child in element.iter():
        if child is element:
            continue

        if len(child) == 0:
            return child

    return None


def _format_value(value) -> str:
    if value is None:
        return "0"

    if isinstance(value, bool):
        return str(value).lower()

    return str(value)