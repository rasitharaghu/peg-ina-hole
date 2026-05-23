import logging
import random
import time
from datetime import datetime, timezone

from opcua import Server


RUNTIME_KEYWORDS = [
    "temperature",
    "pressure",
    "torque",
    "angle",
    "rotation",
    "speed",
    "status",
    "fault",
    "error",
    "result",
    "measuredvalue",
    "highlimit",
    "lowlimit",
    "workorder",
    "task",
]


def is_runtime_variable(name: str) -> bool:
    name = name.lower()
    return any(keyword in name for keyword in RUNTIME_KEYWORDS)


def make_dynamic_value(name: str, counter: int):
    name = name.lower()

    if "temperature" in name:
        return round(70 + random.uniform(-2, 2), 2)

    if "pressure" in name:
        return round(random.uniform(5.8, 6.2), 2)

    if "torque" in name:
        return round(random.uniform(10.5, 12.0), 2)

    if "angle" in name:
        return round(random.uniform(0, 360), 2)

    if "rotation" in name or "speed" in name:
        return random.randint(1450, 1550)

    if "fault" in name or "error" in name:
        return 0 if random.random() > 0.02 else 404

    if "status" in name:
        return "Running"

    if "workorder" in name:
        return f"WO_{1000 + counter}"

    if "task" in name:
        return f"TASK_{counter}"

    if "result" in name:
        return 1

    return round(random.uniform(0, 100), 2)


def browse_runtime_variables(node, runtime_variables):
    try:
        children = node.get_children()
    except Exception:
        return

    for child in children:
        try:
            node_class = child.get_node_class().name
            browse_name = child.get_browse_name().Name

            if node_class == "Variable" and is_runtime_variable(browse_name):
                runtime_variables[browse_name] = child

            browse_runtime_variables(child, runtime_variables)

        except Exception:
            continue


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )

    endpoint = "opc.tcp://0.0.0.0:4840"
    siome5_nodeset_path = "AtlasCopco-Tools.Nodeset2_ToBeWantedSample.xml"

    server = Server()
    server.set_endpoint(endpoint)

    logging.info("Importing SIOME5 NodeSet XML into OPC UA server...")
    server.import_xml(siome5_nodeset_path)

    server.start()
    logging.info("Server started at %s", endpoint)

    runtime_variables = {}

    try:
        objects = server.get_objects_node()
        browse_runtime_variables(objects, runtime_variables)

        logging.info("Runtime variables found: %d", len(runtime_variables))
        logging.info("Runtime variable names: %s", list(runtime_variables.keys()))

        counter = 0

        while True:
            counter += 1

            for name, node in runtime_variables.items():
                try:
                    value = make_dynamic_value(name, counter)
                    node.set_value(value)
                except Exception as exc:
                    logging.debug("Could not update %s: %s", name, exc)

            logging.info(
                "Updated %d runtime variables at %s",
                len(runtime_variables),
                datetime.now(timezone.utc).isoformat(),
            )

            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("Stopping server...")

    finally:
        server.stop()


if __name__ == "__main__":
    main()