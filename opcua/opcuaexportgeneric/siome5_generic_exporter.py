import json
from lxml import etree
from opcua import Client

UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"


def export_siome5_with_mapping(endpoint_url, template_path, mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    client = Client(endpoint_url)
    client.connect()

    try:
        tree = etree.parse(template_path)
        root = tree.getroot()

        for var_node in root.findall(f"{{{UA_NS}}}UAVariable"):
            siome5_node_id = var_node.get("NodeId")

            if siome5_node_id not in mapping:
                continue

            source_node_id = mapping[siome5_node_id]["source_node_id"]

            try:
                live_value = client.get_node(source_node_id).get_value()
            except Exception:
                continue

            update_value(var_node, live_value)

        return etree.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True
        )

    finally:
        client.disconnect()


def update_value(var_node, live_value):
    old_value = var_node.find(f"{{{UA_NS}}}Value")
    if old_value is not None:
        var_node.remove(old_value)

    value_node = etree.SubElement(var_node, f"{{{UA_NS}}}Value")
    data_type = var_node.get("DataType", "String")

    if "Double" in data_type:
        child = etree.SubElement(value_node, f"{{{UAX_NS}}}Double")
        child.text = str(float(live_value))

    elif "Boolean" in data_type:
        child = etree.SubElement(value_node, f"{{{UAX_NS}}}Boolean")
        child.text = str(bool(live_value)).lower()

    elif "DateTime" in data_type or "UtcTime" in data_type:
        child = etree.SubElement(value_node, f"{{{UAX_NS}}}DateTime")
        child.text = str(live_value)

    elif "Int32" in data_type or "Integer" in data_type:
        child = etree.SubElement(value_node, f"{{{UAX_NS}}}Int32")
        child.text = str(int(live_value))

    else:
        child = etree.SubElement(value_node, f"{{{UAX_NS}}}String")
        child.text = str(live_value)