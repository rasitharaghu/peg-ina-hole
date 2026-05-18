from pathlib import Path
from lxml import etree


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"

NS = {
    "ua": UA_NS,
    "uax": UAX_NS,
}


def generate_siome5_xml_from_db_nodes(nodes, template_path: str) -> bytes:
    """
    Final XML:
    - SIOME5 structure/model comes from template
    - Node metadata comes from DB if matching node exists
    - Value comes from DB latest reading if available
    - Missing value becomes 0
    """

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(template_path, parser)
    root = tree.getroot()

    db_nodes_by_label = _build_db_node_lookup(nodes)

    for variable in root.xpath(".//ua:UAVariable", namespaces=NS):
        template_browse_name = variable.get("BrowseName", "")
        template_field_name = _clean_browse_name(template_browse_name)

        db_node = db_nodes_by_label.get(template_field_name)

        if db_node is not None:
            _update_metadata_from_db(variable, db_node)
            value = _get_latest_node_value(db_node)
            if value is None:
                value = 0
        else:
            value = 0

        _set_or_update_value(variable, value)

    return etree.tostring(
        tree,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )


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
    """
    Examples:
        '4:Temperature' -> 'Temperature'
        'Temperature'   -> 'Temperature'
        'Robot.Torque'  -> 'Torque'
    """

    if not name:
        return ""

    name = str(name).strip()

    if ":" in name:
        name = name.split(":", 1)[-1]

    if "." in name:
        name = name.split(".")[-1]

    return name.strip()


def _update_metadata_from_db(variable, db_node):
    """
    Updates metadata from database node.
    SIOME5 tree/hierarchy is still preserved.
    """

    db_node_id = getattr(db_node, "node_id", None)
    db_label = getattr(db_node, "label", None)
    db_data_type = getattr(db_node, "data_type", None)
    db_parent_node_id = getattr(db_node, "parent_node_id", None)

    if db_node_id:
        variable.set("NodeId", str(db_node_id))

    if db_label:
        variable.set("BrowseName", str(db_label))

        display_name_elements = variable.xpath("./ua:DisplayName", namespaces=NS)
        if display_name_elements:
            display_name_elements[0].text = str(db_label)

    if db_data_type:
        variable.set("DataType", str(db_data_type))

    # Optional: update ParentNodeId only if your DB parent_node_id is reliable.
    # If you want SIOME5 hierarchy to remain strictly template-based, comment this block.
    if db_parent_node_id:
        variable.set("ParentNodeId", str(db_parent_node_id))


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


def _set_or_update_value(variable, value):
    data_type = variable.get("DataType", "String")

    value_elements = variable.xpath("./ua:Value", namespaces=NS)

    if value_elements:
        value_element = value_elements[0]
    else:
        value_element = etree.SubElement(variable, f"{{{UA_NS}}}Value")

    leaf = _find_first_leaf(value_element)

    if leaf is not None:
        leaf.text = _format_value(value, data_type)
        return

    child_tag = _get_uax_tag_for_datatype(data_type)
    child = etree.SubElement(value_element, f"{{{UAX_NS}}}{child_tag}")
    child.text = _format_value(value, data_type)


def _find_first_leaf(element):
    for child in element.iter():
        if child is element:
            continue

        if len(child) == 0:
            return child

    return None


def _get_uax_tag_for_datatype(data_type: str) -> str:
    mapping = {
        "Boolean": "Boolean",
        "String": "String",
        "Double": "Double",
        "Float": "Float",
        "Int16": "Int16",
        "Int32": "Int32",
        "Int64": "Int64",
        "UInt16": "UInt16",
        "UInt32": "UInt32",
        "UInt64": "UInt64",
        "Byte": "Byte",
        "SByte": "SByte",
        "DateTime": "DateTime",
    }

    return mapping.get(data_type, "String")


def _format_value(value, data_type: str) -> str:
    if value is None:
        value = 0

    if data_type == "Boolean":
        return str(bool(value)).lower()

    return str(value)