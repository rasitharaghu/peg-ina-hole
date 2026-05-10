import xml.etree.ElementTree as ET


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"

NS = {
    "ua": UA_NS,
}

ET.register_namespace("", UA_NS)
ET.register_namespace("uax", UAX_NS)


def generate_nodeset_xml_from_nodes(nodes, template_path: str) -> bytes:
    tree = ET.parse(template_path)
    root = tree.getroot()

    runtime_values = _extract_runtime_values_from_nodes(nodes)

    for variable in root.findall(".//ua:UAVariable", NS):
        browse_name = variable.attrib.get("BrowseName", "")
        field_name = browse_name.split(":")[-1]

        if field_name not in runtime_values:
            continue

        value_element = variable.find("ua:Value", NS)

        if value_element is None:
            continue

        leaf = _find_first_leaf(value_element)

        if leaf is not None:
            leaf.text = _format_value(runtime_values[field_name])

    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
    )


def _find_first_leaf(element):
    for child in element.iter():
        if child is element:
            continue

        if len(list(child)) == 0:
            return child

    return None


def _extract_runtime_values_from_nodes(nodes) -> dict:
    values = {}

    for node in nodes:
        field_name = getattr(node, "label", None)

        if not field_name:
            field_name = getattr(node, "node_id", "")

        field_name = str(field_name).split(".")[-1].strip()

        if not field_name:
            continue

        latest_reading = node.readings.first()

        if latest_reading is None:
            continue

        if hasattr(latest_reading, "typed_value"):
            values[field_name] = latest_reading.typed_value
        else:
            values[field_name] = latest_reading.value

    return values


def _format_value(value) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value).lower()

    return str(value)