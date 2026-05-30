import json
from pathlib import Path
from difflib import SequenceMatcher
from lxml import etree
from opcua import Client

UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"


def export_siome5_with_mapping(endpoint_url, template_path, mapping):
    mapping_dict = load_mapping(mapping)

    client = Client(endpoint_url)
    client.connect()

    try:
        tree = etree.parse(template_path)
        root = tree.getroot()

        for var_node in root.findall(f"{{{UA_NS}}}UAVariable"):
            siome5_node_id = var_node.get("NodeId")

            if siome5_node_id not in mapping_dict:
                continue

            source_node_id = mapping_dict[siome5_node_id]["source_node_id"]

            try:
                live_value = client.get_node(source_node_id).get_value()
                update_value(var_node, live_value)
            except Exception:
                continue

        return etree.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True,
        )

    finally:
        client.disconnect()


def load_mapping(mapping):
    if isinstance(mapping, dict):
        return mapping

    with open(Path(mapping), "r", encoding="utf-8") as f:
        return json.load(f)


def auto_create_mapping_from_server(endpoint_url, template_path, save_path=None):
    siome5_vars = extract_siome5_variables(template_path)
    source_vars = browse_server_variables(endpoint_url)

    generated_mapping = {}

    for siome5_var in siome5_vars:
        best_match = None
        best_score = 0.0

        for source_var in source_vars:
            score = similarity(
                siome5_var["browse_name_clean"],
                source_var["browse_name_clean"],
            )

            if score > best_score:
                best_score = score
                best_match = source_var

        if best_match and best_score >= 0.75:
            generated_mapping[siome5_var["node_id"]] = {
                "source_node_id": best_match["node_id"],
                "siome5_browse_name": siome5_var["browse_name"],
                "source_browse_name": best_match["browse_name"],
                "match_score": round(best_score, 3),
            }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(generated_mapping, f, indent=2)

    return generated_mapping


def extract_siome5_variables(template_path):
    tree = etree.parse(template_path)
    root = tree.getroot()

    variables = []

    for var in root.findall(f"{{{UA_NS}}}UAVariable"):
        browse_name = var.get("BrowseName", "")
        variables.append({
            "node_id": var.get("NodeId", ""),
            "browse_name": browse_name,
            "browse_name_clean": clean_browse_name(browse_name),
            "data_type": var.get("DataType", "String"),
        })

    return variables


def browse_server_variables(endpoint_url):
    client = Client(endpoint_url)
    client.connect()

    try:
        variables = []
        recursive_browse(client.get_objects_node(), variables)
        return variables
    finally:
        client.disconnect()


def recursive_browse(node, variables):
    try:
        children = node.get_children()
    except Exception:
        return

    for child in children:
        try:
            node_class = child.get_node_class().name
            browse_name = child.get_browse_name().Name
            node_id = child.nodeid.to_string()

            if node_class == "Variable":
                variables.append({
                    "node_id": node_id,
                    "browse_name": browse_name,
                    "browse_name_clean": clean_browse_name(browse_name),
                })

            recursive_browse(child, variables)

        except Exception:
            continue


def clean_browse_name(name):
    if not name:
        return ""

    if ":" in name:
        name = name.split(":", 1)[1]

    return (
        name.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
    )


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def update_value(var_node, live_value):
    old_value = var_node.find(f"{{{UA_NS}}}Value")
    if old_value is not None:
        var_node.remove(old_value)

    value_node = etree.SubElement(var_node, f"{{{UA_NS}}}Value")
    data_type = var_node.get("DataType", "String")

    try:
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

    except Exception:
        child = etree.SubElement(value_node, f"{{{UAX_NS}}}String")
        child.text = str(live_value)