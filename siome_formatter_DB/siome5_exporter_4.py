from datetime import datetime, timezone
from lxml import etree
from opcua import Client


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
NS = {"ua": UA_NS}

SIMPLE_TYPES = {
    "Boolean", "String", "Double", "Float",
    "Int16", "Int32", "Int64",
    "UInt16", "UInt32", "UInt64",
    "Byte", "SByte", "DateTime",
}


def generate_siome5_xml_from_db_nodes(nodes, template_path: str, endpoint_url: str | None = None) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(template_path, parser)
    root = tree.getroot()

    root.set("LastModified", _current_utc_timestamp())

    db_nodes_by_label = _build_db_node_lookup(nodes)
    live_client = None

    if endpoint_url:
        try:
            live_client = Client(endpoint_url)
            live_client.connect()
        except Exception:
            live_client = None

    try:
        for variable in root.xpath(".//ua:UAVariable", namespaces=NS):
            data_type = variable.get("DataType", "")

            if data_type not in SIMPLE_TYPES:
                continue

            template_field_name = _clean_browse_name(variable.get("BrowseName", ""))
            db_node = db_nodes_by_label.get(template_field_name)

            if db_node is not None:
                value = _get_latest_node_value(db_node)

                if value is None and live_client is not None:
                    value = _read_live_value(live_client, db_node)

                if value is None:
                    value = 0
            else:
                value = 0

            _update_existing_simple_value_only(variable, value)

    finally:
        if live_client is not None:
            try:
                live_client.disconnect()
            except Exception:
                pass

    xml_body = etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=False,
        pretty_print=False,
    ).decode("utf-8")

    final_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body
    return final_xml.encode("utf-8")


def _read_live_value(client, db_node):
    try:
        node_id = getattr(db_node, "node_id", None)
        if not node_id:
            return None

        ua_node = client.get_node(str(node_id))
        return ua_node.get_value()

    except Exception:
        return None


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
    value_elements = variable.xpath("./ua:Value", namespaces=NS)

    if not value_elements:
        return

    value_element = value_elements[0]
    children = list(value_element)

    if not children:
        return

    first_child = children[0]

    # Only update direct primitive value, not nested ExtensionObject metadata
    if len(first_child) == 0:
        first_child.text = _format_value(value, variable.get("DataType", ""))


def _format_value(value, data_type: str) -> str:
    if value is None:
        return "0"

    if data_type == "Boolean":
        return str(bool(value)).lower()

    return str(value)