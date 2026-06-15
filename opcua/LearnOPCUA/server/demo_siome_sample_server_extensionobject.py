import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import json

from opcua import Server, ua


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("atlascopco_test_server")

ENDPOINT = "opc.tcp://0.0.0.0:4840/siome_sample/server/"
SERVER_NAME = "AtlasCopco SIOME Test OPC UA Server"

ROOT_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_XML = ROOT_DIR / "AtlasCopco-Tools.Nodeset2_ToBeWantedSample.xml"

UA_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"
NS = {"ua": UA_NS, "uax": UAX_NS}


PRIMITIVE_DEFAULTS = {
    "Boolean": False,
    "Byte": 0,
    "SByte": 0,
    "Int16": 0,
    "UInt16": 0,
    "Int32": 0,
    "UInt32": 0,
    "UInteger": 0,
    "Int64": 0,
    "UInt64": 0,
    "Float": 0.0,
    "Double": 0.0,
    "Duration": 0.0,
    "String": "",
    "TrimmedString": "",
    "DateTime": datetime.now(timezone.utc),
    "UtcTime": datetime.now(timezone.utc),
    "LocalizedText": ua.LocalizedText(""),
    "EUInformation": "unit",
    "EnumValueType": 0,
    "StepTraceDataType": "0,1,2,3",
    "JoiningTraceDataType": "JoiningTrace",
}


VARIANT_TYPES = {
    "Boolean": ua.VariantType.Boolean,
    "Byte": ua.VariantType.Byte,
    "SByte": ua.VariantType.SByte,
    "Int16": ua.VariantType.Int16,
    "UInt16": ua.VariantType.UInt16,
    "Int32": ua.VariantType.Int32,
    "UInt32": ua.VariantType.UInt32,
    "UInteger": ua.VariantType.UInt32,
    "Int64": ua.VariantType.Int64,
    "UInt64": ua.VariantType.UInt64,
    "Float": ua.VariantType.Float,
    "Double": ua.VariantType.Double,
    "Duration": ua.VariantType.Double,
    "String": ua.VariantType.String,
    "TrimmedString": ua.VariantType.String,
    "DateTime": ua.VariantType.DateTime,
    "UtcTime": ua.VariantType.DateTime,
    "LocalizedText": ua.VariantType.LocalizedText,
    "EUInformation": ua.VariantType.String,
    "EnumValueType": ua.VariantType.Int64,
    "StepTraceDataType": ua.VariantType.String,
    "JoiningTraceDataType": ua.VariantType.String,
}


RUNTIME_KEYWORDS = [
    "temperature", "pressure", "torque", "angle", "rotation", "speed",
    "status", "fault", "error", "result", "measuredvalue",
    "highlimit", "lowlimit", "workorder", "task",
    "trace", "jobid", "resultid",
]


def clean_browse_name(browse_name: str) -> str:
    if not browse_name:
        return "Unnamed"
    if ":" in browse_name:
        browse_name = browse_name.split(":", 1)[1]
    return browse_name.strip() or "Unnamed"


def make_variant(data_type, value):
    variant_type = VARIANT_TYPES.get(data_type, ua.VariantType.String)
    if value is None:
        value = PRIMITIVE_DEFAULTS.get(data_type, "")
    return ua.Variant(value, variant_type)


def is_runtime_variable(name: str) -> bool:
    return any(k in name.lower() for k in RUNTIME_KEYWORDS)


def make_dynamic_value(name: str, data_type: str, counter: int):
    name_l = name.lower()

    if "temperature" in name_l:
        return round(70 + random.uniform(-2, 2), 2)
    if "maxtorque" in name_l:
        return round(random.uniform(20.0, 25.0), 2)
    if "torque" in name_l:
        return round(random.uniform(10.5, 12.5), 2)
    if "angle" in name_l:
        return round(random.uniform(0, 360), 2)
    if "speed" in name_l:
        return random.randint(1450, 1550)
    if "status" in name_l:
        return random.choice(["Running", "Idle", "Completed"])
    if "error" in name_l:
        return 0
    if "resultid" in name_l:
        return f"RESULT_{counter:04d}"
    if "jobid" in name_l:
        return f"JOB_{counter:04d}"
    if "workorder" in name_l:
        return f"WO_{1000 + counter}"
    if "task" in name_l:
        return f"TASK_{counter % 10}"
    if "evaluation" in name_l or data_type == "EnumValueType":
        return random.choice([0, 1, 2])
    if "trace" in name_l:
        return ",".join(str(round(random.uniform(0, 20), 2)) for _ in range(5))

    return None


def read_template_value(xml_node, data_type):
    value_el = xml_node.find("ua:Value", NS)
    if value_el is None:
        return PRIMITIVE_DEFAULTS.get(data_type, "")

    children = list(value_el)
    if not children:
        return PRIMITIVE_DEFAULTS.get(data_type, "")

    child = children[0]

    if child.tag.endswith("ExtensionObject") or child.tag.endswith("ListOfExtensionObject"):
        return PRIMITIVE_DEFAULTS.get(data_type, "")

    text = child.text
    if text is None:
        return PRIMITIVE_DEFAULTS.get(data_type, "")

    try:
        if data_type in {"Double", "Float", "Duration"}:
            return float(text)
        if data_type in {"Byte", "SByte", "Int16", "UInt16", "Int32", "UInt32", "UInteger", "Int64", "UInt64"}:
            return int(text)
        if data_type == "Boolean":
            return text.lower() == "true"
        if data_type in {"DateTime", "UtcTime"}:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        if data_type == "LocalizedText":
            return ua.LocalizedText(text)
        return text
    except Exception:
        return PRIMITIVE_DEFAULTS.get(data_type, "")


def get_display_name(xml_node):
    dn = xml_node.find("ua:DisplayName", NS)
    if dn is not None and dn.text:
        return dn.text
    return clean_browse_name(xml_node.attrib.get("BrowseName", ""))


def get_description(xml_node):
    desc = xml_node.find("ua:Description", NS)
    if desc is not None and desc.text:
        return desc.text
    return None


def build_server_from_template(server: Server, template_path: Path):
    tree = ET.parse(template_path)
    root = tree.getroot()

    idx = server.register_namespace("http://ab.com/IJT/AFastening")
    objects = server.get_objects_node()

    created = {
        "i=85": objects,
        "ns=0;i=85": objects,
    }

    runtime_nodes = {}

    nodes = []
    nodes.extend(root.findall("ua:UAObject", NS))
    nodes.extend(root.findall("ua:UAVariable", NS))

    pending = list(nodes)

    for _ in range(30):
        remaining = []

        for xml_node in pending:
            nodeid = xml_node.attrib.get("NodeId", "")
            parent_nodeid = xml_node.attrib.get("ParentNodeId", "")

            if nodeid in created:
                continue

            parent = created.get(parent_nodeid)
            if parent is None:
                remaining.append(xml_node)
                continue

            browse_name = clean_browse_name(xml_node.attrib.get("BrowseName", ""))
            display_name = get_display_name(xml_node)
            description = get_description(xml_node)

            try:
                if xml_node.tag.endswith("UAObject"):
                    opc_node = parent.add_object(idx, browse_name)

                elif xml_node.tag.endswith("UAVariable"):
                    data_type = xml_node.attrib.get("DataType", "String")
                    value = read_template_value(xml_node, data_type)
                    opc_node = parent.add_variable(idx, browse_name, make_variant(data_type, value))
                    opc_node.set_writable()

                    if is_runtime_variable(browse_name) or is_runtime_variable(display_name):
                        runtime_nodes[nodeid] = (display_name, data_type, opc_node)

                else:
                    continue

                try:
                    opc_node.set_display_name(display_name)
                except Exception:
                    pass

                if description:
                    try:
                        opc_node.set_description(description)
                    except Exception:
                        pass

                created[nodeid] = opc_node

            except Exception as exc:
                logger.debug("Could not create %s: %s", browse_name, exc)
                remaining.append(xml_node)

        if len(remaining) == len(pending):
            break
        pending = remaining

    logger.info("Created %d nodes from template", len(created))
    logger.info("Skipped %d unresolved nodes", len(pending))

    return objects, runtime_nodes



def structured_json(data_type, type_id, **fields):
    payload = {
        "__DataType": data_type,
        "__TypeId": type_id,
    }
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def eu_information_value(unit_id, text, description, namespace_uri="http://www.opcfoundation.org/UA/units/un/cefact"):
    return structured_json(
        "EUInformation",
        "i=888",
        NamespaceUri=namespace_uri,
        UnitId=int(unit_id),
        DisplayName={"Text": text},
        Description={"Text": description},
    )


def enum_value_type_value(value, display_text, description_text=""):
    return structured_json(
        "EnumValueType",
        "i=7616",
        Value=int(value),
        DisplayName={"Text": display_text},
        Description={"Text": description_text},
    )


# def step_trace_value(trace_id, values, engineering_unit="Nm"):
#     return structured_json(
#         "StepTraceDataType",
#         "ns=4;i=5013",
#         StepTraceId=str(trace_id),
#         SamplingInterval=10.0,
#         NumberOfTracePoints=len(values),
#         EngineeringUnits=engineering_unit,
#         Values=",".join(str(v) for v in values),
#     )

def step_trace_value(trace_id, values, engineering_unit="Nm"):
    return structured_json(
        "StepTraceDataType",
        "ns=4;i=5069",
        EncodingMask=3,
        NumberOfTracePoints=len(values),
        SamplingInterval=10.0,
        StartTimeOffset=0,
    )


# def joining_trace_value(trace_id):
#     return structured_json(
#         "JoiningTraceDataType",
#         "ns=4;i=5012",
#         TraceId=str(trace_id),
#         TorqueTrace="10.1,10.5,10.8,11.0",
#         AngleTrace="0,25,50,75,100",
#         CurrentTrace="1.2,1.4,1.6,1.8",
#     )

def joining_trace_value(trace_id):
    return structured_json(
        "JoiningTraceDataType",
        "ns=4;i=5066",
    )


# def add_extra_test_nodes(objects, server):
#     idx = server.register_namespace("http://dummy.test/siome/custom-types")

#     test_root = objects.add_object(idx, "SIOME_DataType_Test")

#     # TrimmedString behaves like String in runtime, but client can still export
#     # DataType="TrimmedString" if JSON metadata is provided.
#     test_root.add_variable(
#         idx,
#         "ResultId",
#         make_variant("String", structured_json("TrimmedString", "", Value="RESULT_0001")),
#     ).set_writable()

#     test_root.add_variable(
#         idx,
#         "JobId",
#         make_variant("String", structured_json("TrimmedString", "", Value="JOB_0001")),
#     ).set_writable()

#     # EnumValueType as JSON. The client converts this to:
#     # <uax:ExtensionObject><uax:TypeId><uax:Identifier>i=7616</...>
#     test_root.add_variable(
#         idx,
#         "ResultEvaluation",
#         make_variant("String", enum_value_type_value(1, "OK", "Result is OK")),
#     ).set_writable()

#     test_root.add_variable(
#         idx,
#         "DesignType",
#         make_variant("String", enum_value_type_value(2, "RIGHT_ANGLE", "Right angle tool")),
#     ).set_writable()

#     test_root.add_variable(
#         idx,
#         "DriveMethod",
#         make_variant("String", enum_value_type_value(3, "ELECTRIC", "Electric drive")),
#     ).set_writable()

#     # EUInformation as JSON. This lets the client generate the exact
#     # <uax:EUInformation> sub-tree used by SIOME.
#     test_root.add_variable(
#         idx,
#         "EngineeringUnits",
#         make_variant("String", eu_information_value(5066068, "N·m", "newton metre")),
#     ).set_writable()

#     # StepTraceDataType / JoiningTraceDataType as structured JSON.
#     test_root.add_variable(
#         idx,
#         "TorqueTrace",
#         make_variant("String", step_trace_value("TorqueTrace_001", [10.1, 10.5, 10.8, 11.0], "N·m")),
#     ).set_writable()

#     test_root.add_variable(
#         idx,
#         "AngleTrace",
#         make_variant("String", step_trace_value("AngleTrace_001", [0, 25, 50, 75, 100], "deg")),
#     ).set_writable()

#     test_root.add_variable(
#         idx,
#         "CurrentTrace",
#         make_variant("String", step_trace_value("CurrentTrace_001", [1.2, 1.4, 1.6, 1.8], "A")),
#     ).set_writable()

#     test_root.add_variable(
#         idx,
#         "Trace",
#         make_variant("String", joining_trace_value("Trace_001")),
#     ).set_writable()

#     logger.info("Added extra ExtensionObject datatype stress-test nodes")

def add_extra_test_nodes(objects, server):
    datatype_nodeids = create_custom_datatypes(server)

    idx = server.register_namespace("http://dummy.test/siome/runtime")
    test_root = objects.add_object(idx, "SIOME_DataType_Test")

    result_id = test_root.add_variable(idx, "ResultId", "RESULT_0001")
    set_variable_datatype(result_id, datatype_nodeids["TrimmedString"])
    result_id.set_writable()

    job_id = test_root.add_variable(idx, "JobId", "JOB_0001")
    set_variable_datatype(job_id, datatype_nodeids["TrimmedString"])
    job_id.set_writable()

    result_eval = test_root.add_variable(
        idx,
        "ResultEvaluation",
        enum_value_type_value(0, "OK", "Result OK")
    )
    set_variable_datatype(result_eval, datatype_nodeids["EnumValueType"])
    result_eval.set_writable()

    engineering_units = test_root.add_variable(
        idx,
        "EngineeringUnits",
        eu_information_value(
            unit_id=4408652,
            text="N·m",
            description="newton metre"
        )
    )
    set_variable_datatype(engineering_units, datatype_nodeids["EUInformation"])
    engineering_units.set_writable()

    torque_trace = test_root.add_variable(
        idx,
        "TorqueTrace",
        step_trace_value("TorqueTrace_001", [10.1, 10.5, 10.8, 11.0], "N·m")
    )
    set_variable_datatype(torque_trace, datatype_nodeids["StepTraceDataType"])
    torque_trace.set_writable()

    angle_trace = test_root.add_variable(
        idx,
        "AngleTrace",
        step_trace_value("AngleTrace_001", [0, 25, 50, 75, 100], "deg")
    )
    set_variable_datatype(angle_trace, datatype_nodeids["StepTraceDataType"])
    angle_trace.set_writable()

    # trace_content = test_root.add_variable(
    #     idx,
    #     "StepTraceContent",
    #     structured_json(
    #         "TraceContentDataType",
    #         "ns=4;i=5072",
    #         EncodingMask=24,
    #         PhysicalQuantity=0,
    #         EngineeringUnits={"UnitId": 0}
    #     )
    # )

    trace_content = test_root.add_variable(
        idx,
        "StepTraceContent",
        structured_json(
            "TraceContentDataType",
            "ns=4;i=5072",
            EncodingMask=24,
            PhysicalQuantity=0,
            EngineeringUnits={"UnitId": 0}
        )
    )
    set_variable_datatype(trace_content, datatype_nodeids["TraceContentDataType"])
    trace_content.set_writable()

    joining_trace = test_root.add_variable(
        idx,
        "Trace",
        joining_trace_value("Trace_001")
    )
    set_variable_datatype(joining_trace, datatype_nodeids["JoiningTraceDataType"])
    joining_trace.set_writable()

    logger.info("Added datatype nodes and ExtensionObject test variables")

def set_variable_datatype(var_node, datatype_nodeid):
    var_node.set_attribute(
        ua.AttributeIds.DataType,
        ua.DataValue(ua.Variant(datatype_nodeid, ua.VariantType.NodeId))
    )


def create_custom_datatypes(server):
    base_data_type = server.get_node(ua.NodeId(24, 0))  # BaseDataType

    ijt_idx = server.register_namespace("http://opcfoundation.org/UA/IJT/Base/")

    joining_trace_dt = base_data_type.add_data_type(ijt_idx, "JoiningTraceDataType")
    step_trace_dt = base_data_type.add_data_type(ijt_idx, "StepTraceDataType")
    trace_content_dt = base_data_type.add_data_type(ijt_idx, "TraceContentDataType")
    trimmed_string_dt = base_data_type.add_data_type(ijt_idx, "TrimmedString")

    return {
        "JoiningTraceDataType": joining_trace_dt.nodeid,
        "StepTraceDataType": step_trace_dt.nodeid,
        "TraceContentDataType": trace_content_dt.nodeid,
        "TrimmedString": trimmed_string_dt.nodeid,
        "EUInformation": ua.NodeId(887, 0),
        "EnumValueType": ua.NodeId(7594, 0),
    }

def main():
    if not TEMPLATE_XML.exists():
        raise FileNotFoundError(f"Template XML not found: {TEMPLATE_XML}")

    server = Server()
    server.set_endpoint(ENDPOINT)
    server.set_server_name(SERVER_NAME)

    objects, runtime_nodes = build_server_from_template(server, TEMPLATE_XML)
    add_extra_test_nodes(objects, server)

    server.start()
    logger.info("Server started at %s", ENDPOINT)
    logger.info("Runtime variables found: %d", len(runtime_nodes))

    counter = 0

    try:
        while True:
            counter += 1

            for _, (name, data_type, node) in runtime_nodes.items():
                value = make_dynamic_value(name, data_type, counter)
                if value is None:
                    continue

                try:
                    node.set_value(make_variant(data_type, value))
                except Exception as exc:
                    logger.debug("Could not update %s: %s", name, exc)

            logger.info("Updated %d runtime variables at %s", len(runtime_nodes), datetime.now(timezone.utc).isoformat())
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info("Shutdown requested")

    finally:
        server.stop()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()