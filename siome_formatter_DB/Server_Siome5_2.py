import random
import time
import logging
from datetime import datetime
from lxml import etree
from opcua import Server


UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
NS = {"ua": UA_NS}

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


def clean_browse_name(name: str) -> str:
    if not name:
        return ""

    name = str(name).strip()

    if ":" in name:
        name = name.split(":", 1)[-1]

    if "." in name:
        name = name.split(".")[-1]

    return name.strip()


def read_simple_siome5_variables(template_path: str):
    tree = etree.parse(template_path)

    variables = []

    for variable in tree.xpath(".//ua:UAVariable", namespaces=NS):
        browse_name = variable.get("BrowseName", "")
        data_type = variable.get("DataType", "")

        field_name = clean_browse_name(browse_name)

        if not field_name:
            continue

        if data_type not in SIMPLE_TYPES:
            continue

        variables.append({
            "field_name": field_name,
            "data_type": data_type,
        })

    # Remove duplicate field names
    unique = {}
    for item in variables:
        unique[item["field_name"]] = item

    return list(unique.values())


def default_value_for_type(data_type: str):
    if data_type in {"Double", "Float"}:
        return 0.0

    if data_type in {
        "Int16", "Int32", "Int64",
        "UInt16", "UInt32", "UInt64",
        "Byte", "SByte"
    }:
        return 0

    if data_type == "Boolean":
        return False

    if data_type == "DateTime":
        return datetime.utcnow()

    return "0"


def dynamic_value(field_name: str, data_type: str, counter: int):
    name = field_name.lower()

    if data_type in {"Double", "Float"}:
        if "temperature" in name:
            return round(70 + random.uniform(-2, 2), 2)
        if "pressure" in name:
            return round(random.uniform(5.8, 6.2), 2)
        if "torque" in name:
            return round(random.uniform(10.5, 12.0), 2)
        if "angle" in name:
            return round(random.uniform(0, 360), 2)
        if "voltage" in name:
            return round(random.uniform(225, 235), 2)
        if "current" in name:
            return round(random.uniform(4.5, 7.5), 2)
        if "limit" in name:
            return round(random.uniform(0, 100), 2)
        return round(random.uniform(0, 100), 2)

    if data_type in {
        "Int16", "Int32", "Int64",
        "UInt16", "UInt32", "UInt64",
        "Byte", "SByte"
    }:
        if "fault" in name or "error" in name:
            return 0 if random.random() > 0.02 else 404
        if "count" in name:
            return counter
        if "rpm" in name or "rotation" in name:
            return random.randint(1450, 1550)
        return random.randint(0, 100)

    if data_type == "Boolean":
        return random.random() > 0.1

    if data_type == "DateTime":
        return datetime.utcnow()

    if data_type == "String":
        if "status" in name:
            return "Running"
        if "workorder" in name:
            return f"WO_{1000 + counter}"
        if "task" in name:
            return f"TASK_{counter}"
        if "serial" in name:
            return "SN_001"
        return f"{field_name}_{counter}"

    return "0"


def create_siome5_demo_server(
    template_path: str,
    endpoint: str = "opc.tcp://0.0.0.0:4840",
):
    server = Server()
    server.set_endpoint(endpoint)

    uri = "http://examples.freeopcua.github.io"
    idx = server.register_namespace(uri)

    objects = server.get_objects_node()
    root_obj = objects.add_object(idx, "SIOME5_Demo_Server")

    siome_variables = read_simple_siome5_variables(template_path)

    opcua_nodes = {}

    for item in siome_variables:
        field_name = item["field_name"]
        data_type = item["data_type"]

        initial_value = default_value_for_type(data_type)

        try:
            node = root_obj.add_variable(idx, field_name, initial_value)
            node.set_writable()
            opcua_nodes[field_name] = {
                "node": node,
                "data_type": data_type,
            }
        except Exception as exc:
            logging.warning("Could not create node %s: %s", field_name, exc)

    return server, opcua_nodes


def run_demo_loop(server, opcua_nodes, update_interval=1.0):
    logging.info("Starting SIOME5 dynamic simulation loop")

    counter = 0

    try:
        while True:
            counter += 1

            for field_name, item in opcua_nodes.items():
                node = item["node"]
                data_type = item["data_type"]

                value = dynamic_value(field_name, data_type, counter)

                try:
                    node.set_value(value)
                except Exception as exc:
                    logging.warning("Failed updating %s: %s", field_name, exc)

            logging.info("Updated %d SIOME5 simple nodes", len(opcua_nodes))

            time.sleep(update_interval)

    except KeyboardInterrupt:
        logging.info("Simulation interrupted")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    template_path = "AtlasCopco-Tools.Nodeset2_ToBeWantedSample.xml"
    endpoint = "opc.tcp://0.0.0.0:4840"

    server, opcua_nodes = create_siome5_demo_server(
        template_path=template_path,
        endpoint=endpoint,
    )

    try:
        server.start()
        logging.info("SIOME5 OPC UA demo server started at %s", endpoint)
        logging.info("Created %d dynamic nodes", len(opcua_nodes))

        run_demo_loop(
            server=server,
            opcua_nodes=opcua_nodes,
            update_interval=1.0,
        )

    finally:
        logging.info("Stopping OPC UA server")
        server.stop()


if __name__ == "__main__":
    main()