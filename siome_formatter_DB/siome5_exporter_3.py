from datetime import datetime, timezone
from lxml import etree


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"

NS = {
    "ua": UA_NS,
}


SIMPLE_TYPES = {
    "Boolean",
    "String",
    "Double",
    "Float",
    "Int16",
    "Int32",
    "Int64",
    "UInt16",
    "UInt32",
    "UInt64",
    "Byte",
    "SByte",
    "DateTime",
}


def generate_siome5_xml_from_db_nodes(nodes, template_path: str) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(template_path, parser)
    root = tree.getroot()

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

        _update_existing_simple_value_only(variable, value)

    xml_body = etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=False,
    ).decode("utf-8")

    final_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body
    return final_xml.encode("utf-8")


def _current_utc_timestamp() -> str:
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


def _update_existing_simple_value_only(variable, value):
    data_type = variable.get("DataType", "")

    # Important:
    # Skip complex SIOME/OPC UA types like EUInformation, LocalizedText,
    # EnumValueType, ExtensionObject, etc.
    if data_type not in SIMPLE_TYPES:
        return

    value_elements = variable.xpath("./ua:Value", namespaces=NS)

    # Important:
    # Do not create new <Value> tags.
    # Only update values that already exist in the SIOME template.
    if not value_elements:
        return

    value_element = value_elements[0]
    leaf = _find_simple_value_leaf(value_element)

    if leaf is not None:
        leaf.text = _format_value(value, data_type)


def _find_simple_value_leaf(value_element):
    """
    Only updates simple structure like:

    <Value>
        <uax:Double>0</uax:Double>
    </Value>

    It will NOT update nested ExtensionObject structures.
    """

    children = list(value_element)

    if not children:
        return None

    first_child = children[0]

    if len(first_child) == 0:
        return first_child

    return None


def _format_value(value, data_type: str) -> str:
    if value is None:
        value = 0

    if data_type == "Boolean":
        return str(bool(value)).lower()

    return str(value)