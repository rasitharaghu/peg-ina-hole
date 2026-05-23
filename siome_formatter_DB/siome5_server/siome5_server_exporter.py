from datetime import datetime, timezone

from lxml import etree
from opcua import Client


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"

NSMAP = {
    None: UA_NS,
    "uax": UAX_NS,
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


def export_nodeset_from_server(endpoint_url: str) -> bytes:
    client = Client(endpoint_url)
    client.connect()

    try:
        root = etree.Element(
            f"{{{UA_NS}}}UANodeSet",
            nsmap=NSMAP,
            LastModified=_current_timestamp(),
        )

        _add_basic_header(root)

        objects = client.get_objects_node()
        _browse_and_export(objects, root)

        xml_body = etree.tostring(
            root,
            encoding="utf-8",
            xml_declaration=False,
            pretty_print=True,
        ).decode("utf-8")

        final_xml = '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body

        return final_xml.encode("utf-8")

    finally:
        client.disconnect()


def _browse_and_export(node, xml_root):
    try:
        children = node.get_children()
    except Exception:
        return

    for child in children:
        try:
            node_class = child.get_node_class().name
            browse_name = child.get_browse_name().Name
            display_name = child.get_display_name().Text
            node_id = child.nodeid.to_string()

            if node_class == "Object":
                _add_uaobject(xml_root, child, node_id, browse_name, display_name)

            elif node_class == "Variable":
                _add_uavariable(xml_root, child, node_id, browse_name, display_name)

            _browse_and_export(child, xml_root)

        except Exception:
            continue


def _add_uaobject(xml_root, node, node_id, browse_name, display_name):
    obj = etree.SubElement(
        xml_root,
        f"{{{UA_NS}}}UAObject",
        NodeId=node_id,
        BrowseName=browse_name,
    )

    etree.SubElement(obj, f"{{{UA_NS}}}DisplayName").text = display_name

    _add_references(obj, node)


def _add_uavariable(xml_root, node, node_id, browse_name, display_name):
    data_type = _get_data_type_name(node)

    var = etree.SubElement(
        xml_root,
        f"{{{UA_NS}}}UAVariable",
        NodeId=node_id,
        BrowseName=browse_name,
        DataType=data_type,
    )

    etree.SubElement(var, f"{{{UA_NS}}}DisplayName").text = display_name

    _add_references(var, node)

    value = _get_value(node)
    _add_value(var, data_type, value)


def _add_references(parent_xml, node):
    refs = etree.SubElement(parent_xml, f"{{{UA_NS}}}References")

    try:
        for ref in node.get_references():
            ref_el = etree.SubElement(
                refs,
                f"{{{UA_NS}}}Reference",
                ReferenceType=ref.ReferenceTypeId.to_string(),
                IsForward=str(ref.IsForward).lower(),
            )
            ref_el.text = ref.NodeId.to_string()

    except Exception:
        pass


def _add_value(var_xml, data_type, value):
    if data_type not in SIMPLE_TYPES:
        return

    value_el = etree.SubElement(var_xml, f"{{{UA_NS}}}Value")
    value_child = etree.SubElement(value_el, f"{{{UAX_NS}}}{data_type}")

    value_child.text = _format_value(value, data_type)


def _get_value(node):
    try:
        return node.get_value()
    except Exception:
        return None


def _get_data_type_name(node):
    try:
        data_type_node = node.get_data_type()
        return data_type_node.get_browse_name().Name
    except Exception:
        return "String"


def _format_value(value, data_type: str):
    if value is None:
        return "0"

    if data_type == "Boolean":
        return str(bool(value)).lower()

    if isinstance(value, datetime):
        return value.isoformat()

    return str(value)


def _add_basic_header(root):
    namespace_uris = etree.SubElement(root, f"{{{UA_NS}}}NamespaceUris")
    etree.SubElement(namespace_uris, f"{{{UA_NS}}}Uri").text = (
        "http://airbus.com/IJT/AFastening"
    )

    aliases = etree.SubElement(root, f"{{{UA_NS}}}Aliases")

    alias_map = {
        "Boolean": "i=1",
        "String": "i=12",
        "Double": "i=11",
        "Int32": "i=6",
        "Int64": "i=8",
        "DateTime": "i=13",
        "HasComponent": "i=47",
        "HasProperty": "i=46",
        "HasTypeDefinition": "i=40",
    }

    for alias, value in alias_map.items():
        etree.SubElement(
            aliases,
            f"{{{UA_NS}}}Alias",
            Alias=alias,
        ).text = value


def _current_timestamp():
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(now.microsecond / 1000):03d}Z"