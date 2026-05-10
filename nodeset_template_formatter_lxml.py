from lxml import etree


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"

NS = {
    "ua": UA_NS,
}


def generate_nodeset_xml_from_nodes(nodes, template_path: str) -> bytes:
    parser = etree.XMLParser(
        remove_blank_text=False,
        resolve_entities=False,
    )

    tree = etree.parse(template_path, parser)
    root = tree.getroot()

    runtime_values = _extract_runtime_values_from_nodes(nodes)

    for variable in root.xpath(".//ua:UAVariable", namespaces=NS):
        browse_name = variable.get("BrowseName", "")
        field_name = browse_name.split(":")[-1]

        if field_name not in runtime_values:
            continue

        value_elements = variable.xpath("./ua:Value", namespaces=NS)

        if not value_elements:
            continue

        leaf = _find_first_leaf(value_elements[0])

        if leaf is not None:
            leaf.text = _format_value(runtime_values[field_name])

    return etree.tostring(
        tree,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )


def _find_first_leaf(element):
    for child in element.iter():
        if child is element:
            continue

        if len(child) == 0:
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