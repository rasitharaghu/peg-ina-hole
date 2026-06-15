"""
Improved OPC UA test server for SIOME/NodeSet exporter validation.

Purpose:
- Import OPC UA companion NodeSet2 XML files (DI, AMB, Machinery, Machinery Result, IJT Base, IJT Tightening, Atlas Copco).
- Expose simple runtime variables.
- Expose test nodes for LocalizedText, EUInformation, DateTime, StepTraceDataType, TraceContentDataType.

For custom structures like StepTraceDataType, this server sets the OPC UA DataType
attribute to the imported custom DataType NodeId, while storing structured JSON as
the runtime value. Your exporter can detect this JSON and serialize it as
uax:ExtensionObject / uax:ListOfExtensionObject in the SIOME-compatible XML.
"""

import json
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from opcua import Server, ua


ENDPOINT = "opc.tcp://127.0.0.1:4840/siome_sample/server/"

# Change this to the folder where your NodeSet2 XML files are stored.
XML_DIR = Path(r"C:\Rasitha\LearnOPCUA\xml")

NODESET_FILES = [
    "Opc.Ua.NodeSet2.xml",
    "Opc.Ua.Di.NodeSet2.xml",
    "Opc.Ua.AMB.NodeSet2.xml",
    "Opc.Ua.Machinery.NodeSet2.xml",
    "Opc.Ua.Machinery.Result.NodeSet2.xml",
    "Opc.Ua.Ijt.Base.NodeSet2.xml",
    "Opc.Ua.Ijt.Tightening.NodeSet2.xml",
    "AtlasCopco-Tools.Nodeset2.xml",
]

CUSTOM_TEST_NAMESPACE_URI = "http://learning.opcua.production"


class ProductionLineSimulator:
    def __init__(self):
        self.temperature = 25.0
        self.pressure = 5.0
        self.speed = 0.0
        self.is_running = True
        self.lock = threading.Lock()

    def update_sensors(self):
        with self.lock:
            self.temperature = max(20, min(100, self.temperature + random.uniform(-2, 2)))
            self.pressure = max(1, min(10, self.pressure + random.uniform(-0.5, 0.5)))
            target_speed = 500
            speed_change = (target_speed - self.speed) * 0.1 + random.uniform(-50, 50)
            self.speed = max(0, min(1000, self.speed + speed_change))

    def get_values(self):
        with self.lock:
            return {
                "temperature": round(self.temperature, 2),
                "pressure": round(self.pressure, 2),
                "speed": round(self.speed, 2),
            }


def import_nodesets(server: Server):
    print("[SERVER] Importing NodeSet2 XML files...")
    for file_name in NODESET_FILES:
        path = XML_DIR / file_name
        if not path.exists():
            print(f"[SERVER]   SKIP missing: {path}")
            continue
        try:
            print(f"[SERVER]   Importing: {path}")
            server.import_xml(str(path))
            print(f"[SERVER]   OK: {file_name}")
        except Exception as exc:
            print(f"[SERVER]   WARNING import failed for {file_name}: {exc}")
    print("[SERVER] NodeSet import phase finished\n")


def get_namespace_index(server: Server, namespace_uri: str):
    try:
        return server.get_namespace_index(namespace_uri)
    except Exception:
        pass
    try:
        namespace_array = server.get_namespace_array()
        return namespace_array.index(namespace_uri)
    except Exception:
        return None


def make_nodeid(server: Server, namespace_uri: str, numeric_id: int):
    ns_idx = get_namespace_index(server, namespace_uri)
    if ns_idx is None:
        return None
    return ua.NodeId(numeric_id, ns_idx)


def set_node_attribute(node, attribute_id, value):
    node.set_attribute(attribute_id, ua.DataValue(ua.Variant(value)))


def set_variable_datatype(node, datatype_nodeid):
    if datatype_nodeid is not None:
        set_node_attribute(node, ua.AttributeIds.DataType, datatype_nodeid)


def set_variable_value_rank(node, value_rank):
    set_node_attribute(node, ua.AttributeIds.ValueRank, value_rank)


def set_variable_array_dimensions(node, dimensions):
    set_node_attribute(node, ua.AttributeIds.ArrayDimensions, dimensions)


def make_localized_text(text, locale="en-US"):
    return ua.LocalizedText(text, locale)


def make_eu_information(unit_id, display_name, description):
    eu = ua.EUInformation()
    eu.NamespaceUri = "http://www.opcfoundation.org/UA/units/un/cefact"
    eu.UnitId = int(unit_id)
    eu.DisplayName = ua.LocalizedText(display_name, "en-US")
    eu.Description = ua.LocalizedText(description, "en-US")
    return eu


# def make_step_trace_json(type_id: str):
#     return json.dumps({
#         "__DataType": "StepTraceDataType",
#         "__TypeId": type_id,
#         "__XmlNamespace": "http://opcfoundation.org/UA/IJT/Base/Types.xsd",
#         "__IsArray": True,
#         "__ValueRank": 1,
#         "__ArrayDimensions": [1],
#         "items": [
#             {
#                 "EncodingMask": 3,
#                 "NumberOfTracePoints": 0,
#                 "SamplingInterval": 0,
#                 "StartTimeOffset": 0,
#             }
#         ],
#     })

def make_step_trace_json(type_id: str):
    return json.dumps({
        "__DataType": "StepTraceDataType",
        "__TypeId": type_id,
        "__XmlNamespace": "http://opcfoundation.org/UA/IJT/Base/Types.xsd",
        "EncodingMask": 3,
        "NumberOfTracePoints": 0,
        "SamplingInterval": 0,
        "StartTimeOffset": 0,
    })


def make_trace_content_json(type_id: str):
    return json.dumps({
        "__DataType": "TraceContentDataType",
        "__TypeId": type_id,
        "__XmlNamespace": "http://opcfoundation.org/UA/IJT/Base/Types.xsd",
        "__IsArray": False,
        "EncodingMask": 0,
    })


def make_eu_information_json():
    return json.dumps({
        "__DataType": "EUInformation",
        "__TypeId": "i=888",
        "__XmlNamespace": "http://opcfoundation.org/UA/2008/02/Types.xsd",
        "__IsArray": False,
        "NamespaceUri": "http://www.opcfoundation.org/UA/units/un/cefact",
        "UnitId": 4408652,
        "DisplayName": {"Text": "degC"},
        "Description": {"Text": "degree Celsius"},
    })


# def create_runtime_address_space(server: Server):
#     simulator = ProductionLineSimulator()

#     custom_ns = server.register_namespace(CUSTOM_TEST_NAMESPACE_URI)
#     objects = server.get_objects_node()

#     production_line = objects.add_folder(custom_ns, "ProductionLine")
#     sensors = production_line.add_folder(custom_ns, "Sensors")
#     test = production_line.add_folder(custom_ns, "TestDataTypes")
#     trace = production_line.add_folder(custom_ns, "Trace")

#     temp_node = sensors.add_variable(custom_ns, "Temperature", ua.Variant(25.0, ua.VariantType.Float))
#     pressure_node = sensors.add_variable(custom_ns, "Pressure", ua.Variant(5.0, ua.VariantType.Float))
#     speed_node = sensors.add_variable(custom_ns, "Speed", ua.Variant(0.0, ua.VariantType.Float))

#     for n in (temp_node, pressure_node, speed_node):
#         n.set_writable()

#     sensors.add_variable(custom_ns, "Temperature_Unit", ua.Variant("degC", ua.VariantType.String))
#     sensors.add_variable(custom_ns, "Pressure_Unit", ua.Variant("bar", ua.VariantType.String))
#     sensors.add_variable(custom_ns, "Speed_Unit", ua.Variant("RPM", ua.VariantType.String))

#     localized_status = test.add_variable(
#         custom_ns,
#         "LocalizedStatus",
#         ua.Variant(make_localized_text("Machine running normally", "en-US"), ua.VariantType.LocalizedText),
#     )
#     localized_status.set_writable()

#     last_update = test.add_variable(
#         custom_ns,
#         "LastUpdateTime",
#         ua.Variant(datetime.now(timezone.utc), ua.VariantType.DateTime),
#     )
#     last_update.set_writable()

#     try:
#         eu_info = make_eu_information(4408652, "degC", "degree Celsius")
#         eu_node = test.add_variable(custom_ns, "TemperatureEngineeringUnits", eu_info)
#         eu_node.set_writable()
#         set_variable_datatype(eu_node, ua.NodeId(887, 0))
#     except Exception as exc:
#         print(f"[SERVER] Direct EUInformation failed; using JSON fallback: {exc}")
#         eu_node = test.add_variable(
#             custom_ns,
#             "TemperatureEngineeringUnits",
#             ua.Variant(make_eu_information_json(), ua.VariantType.String),
#         )
#         set_variable_datatype(eu_node, ua.NodeId(887, 0))
#         eu_node.set_writable()

#     ijt_base_uri = "http://opcfoundation.org/UA/IJT/Base/"
#     step_trace_dtype = make_nodeid(server, ijt_base_uri, 3013)
#     trace_content_dtype = make_nodeid(server, ijt_base_uri, 3014)

#     if step_trace_dtype is None:
#         print("[SERVER] WARNING: IJT Base namespace not found. StepTraceDataType DataType will remain String.")
#         step_trace_type_id = "ns=0;i=0"
#     else:
#         # step_trace_type_id = step_trace_dtype.to_string()
#         step_trace_type_id = get_binary_encoding_nodeid(server, step_trace_dtype)

#     if trace_content_dtype is None:
#         print("[SERVER] WARNING: IJT Base namespace not found. TraceContentDataType DataType will remain String.")
#         trace_content_type_id = "ns=0;i=0"
#     else:
#         # trace_content_type_id = trace_content_dtype.to_string()
#         trace_content_type_id = get_binary_encoding_nodeid(server, trace_content_dtype)

#     step_traces = trace.add_variable(
#         custom_ns,
#         "StepTraces",
#         ua.Variant(make_step_trace_json(step_trace_type_id), ua.VariantType.String),
#     )
#     step_traces.set_writable()
#     set_variable_datatype(step_traces, step_trace_dtype)
#     set_variable_value_rank(step_traces, 1)
#     set_variable_array_dimensions(step_traces, [1])

#     trace_content = trace.add_variable(
#         custom_ns,
#         "TraceContent",
#         ua.Variant(make_trace_content_json(trace_content_type_id), ua.VariantType.String),
#     )
#     trace_content.set_writable()
#     set_variable_datatype(trace_content, trace_content_dtype)

#     print("[SERVER] Runtime test address space created")
#     print("[SERVER]   ProductionLine.Sensors.Temperature")
#     print("[SERVER]   ProductionLine.TestDataTypes.LocalizedStatus")
#     print("[SERVER]   ProductionLine.TestDataTypes.TemperatureEngineeringUnits")
#     print(f"[SERVER]   ProductionLine.Trace.StepTraces DataType={step_trace_type_id}")
#     print(f"[SERVER]   ProductionLine.Trace.TraceContent DataType={trace_content_type_id}\n")

#     return simulator, temp_node, pressure_node, speed_node, last_update

def create_runtime_address_space(server: Server):
    simulator = ProductionLineSimulator()

    custom_ns = server.register_namespace(CUSTOM_TEST_NAMESPACE_URI)
    objects = server.get_objects_node()

    production_line = objects.add_folder(custom_ns, "ProductionLine")
    sensors = production_line.add_folder(custom_ns, "Sensors")
    test = production_line.add_folder(custom_ns, "TestDataTypes")
    trace = production_line.add_folder(custom_ns, "Trace")

    # Basic runtime sensor variables
    temp_node = sensors.add_variable(custom_ns, "Temperature", ua.Variant(25.0, ua.VariantType.Float))
    pressure_node = sensors.add_variable(custom_ns, "Pressure", ua.Variant(5.0, ua.VariantType.Float))
    speed_node = sensors.add_variable(custom_ns, "Speed", ua.Variant(0.0, ua.VariantType.Float))

    for n in (temp_node, pressure_node, speed_node):
        n.set_writable()

    sensors.add_variable(custom_ns, "Temperature_Unit", ua.Variant("degC", ua.VariantType.String))
    sensors.add_variable(custom_ns, "Pressure_Unit", ua.Variant("bar", ua.VariantType.String))
    sensors.add_variable(custom_ns, "Speed_Unit", ua.Variant("RPM", ua.VariantType.String))

    # Primitive datatype test variables
    test.add_variable(custom_ns, "BooleanTest", ua.Variant(True, ua.VariantType.Boolean))
    test.add_variable(custom_ns, "Int32Test", ua.Variant(123, ua.VariantType.Int32))
    test.add_variable(custom_ns, "DoubleTest", ua.Variant(12.34, ua.VariantType.Double))
    test.add_variable(custom_ns, "StringTest", ua.Variant("Hello OPC UA", ua.VariantType.String))

    # LocalizedText test
    localized_status = test.add_variable(
        custom_ns,
        "LocalizedStatus",
        ua.Variant(make_localized_text("Machine running normally", "en-US"), ua.VariantType.LocalizedText),
    )
    localized_status.set_writable()

    # DateTime test
    last_update = test.add_variable(
        custom_ns,
        "LastUpdateTime",
        ua.Variant(datetime.now(timezone.utc), ua.VariantType.DateTime),
    )
    last_update.set_writable()

    # EUInformation test
    try:
        eu_info = make_eu_information(4408652, "degC", "degree Celsius")
        eu_node = test.add_variable(custom_ns, "TemperatureEngineeringUnits", eu_info)
        eu_node.set_writable()
        set_variable_datatype(eu_node, ua.NodeId(887, 0))
    except Exception as exc:
        print(f"[SERVER] Direct EUInformation failed; using JSON fallback: {exc}")
        eu_node = test.add_variable(
            custom_ns,
            "TemperatureEngineeringUnits",
            ua.Variant(make_eu_information_json(), ua.VariantType.String),
        )
        set_variable_datatype(eu_node, ua.NodeId(887, 0))
        eu_node.set_writable()

    # IJT custom structure datatype tests
    ijt_base_uri = "http://opcfoundation.org/UA/IJT/Base/"
    step_trace_dtype = make_nodeid(server, ijt_base_uri, 3013)
    trace_content_dtype = make_nodeid(server, ijt_base_uri, 3014)

    if step_trace_dtype is None:
        print("[SERVER] WARNING: IJT Base namespace not found. StepTraceDataType DataType will remain String.")
        step_trace_type_id = "ns=0;i=0"
    else:
        step_trace_type_id = get_binary_encoding_nodeid(server, step_trace_dtype)

    if trace_content_dtype is None:
        print("[SERVER] WARNING: IJT Base namespace not found. TraceContentDataType DataType will remain String.")
        trace_content_type_id = "ns=0;i=0"
    else:
        trace_content_type_id = get_binary_encoding_nodeid(server, trace_content_dtype)

    step_traces = trace.add_variable(
        custom_ns,
        "StepTraces",
        ua.Variant(make_step_trace_json(step_trace_type_id), ua.VariantType.String),
    )
    step_traces.set_writable()

    if step_trace_dtype is not None:
        # set_variable_datatype(step_traces, step_trace_dtype)
        set_variable_datatype(step_traces, ua.NodeId(22, 0))  # Structure / ExtensionObject-related
        set_variable_value_rank(step_traces, 1)
        set_variable_array_dimensions(step_traces, [1])

    trace_content = trace.add_variable(
        custom_ns,
        "TraceContent",
        ua.Variant(make_trace_content_json(trace_content_type_id), ua.VariantType.String),
    )
    trace_content.set_writable()

    if trace_content_dtype is not None:
        # set_variable_datatype(trace_content, trace_content_dtype)
        set_variable_datatype(trace_content, ua.NodeId(22, 0))

    print("[SERVER] Runtime test address space created")
    print("[SERVER]   ProductionLine.Sensors.Temperature")
    print("[SERVER]   ProductionLine.TestDataTypes.BooleanTest")
    print("[SERVER]   ProductionLine.TestDataTypes.Int32Test")
    print("[SERVER]   ProductionLine.TestDataTypes.DoubleTest")
    print("[SERVER]   ProductionLine.TestDataTypes.StringTest")
    print("[SERVER]   ProductionLine.TestDataTypes.LocalizedStatus")
    print("[SERVER]   ProductionLine.TestDataTypes.LastUpdateTime")
    print("[SERVER]   ProductionLine.TestDataTypes.TemperatureEngineeringUnits")
    print(f"[SERVER]   ProductionLine.Trace.StepTraces EncodingId={step_trace_type_id}")
    print(f"[SERVER]   ProductionLine.Trace.TraceContent EncodingId={trace_content_type_id}\n")

    return simulator, temp_node, pressure_node, speed_node, last_update


def get_binary_encoding_nodeid(server, datatype_nodeid):
    if datatype_nodeid is None:
        return ""

    try:
        dt_node = server.get_node(datatype_nodeid)

        for ref in dt_node.get_references(
            refs=ua.ObjectIds.HasEncoding,
            direction=ua.BrowseDirection.Forward,
            includesubtypes=True,
        ):
            enc_node = server.get_node(ref.NodeId)
            browse_name = enc_node.get_browse_name().Name

            if "Binary" in browse_name:
                return ref.NodeId.to_string()

    except Exception:
        pass

    return ""

def main():
    server = Server()
    server.set_endpoint(ENDPOINT)

    print("[SERVER] SIOME / IJT OPC UA Test Server")
    print(f"[SERVER] Endpoint: {ENDPOINT}")
    print(f"[SERVER] XML_DIR: {XML_DIR}\n")

    import_nodesets(server)
    simulator, temp_node, pressure_node, speed_node, last_update = create_runtime_address_space(server)

    server.start()
    print("[SERVER] Server started successfully")
    print("[SERVER] Waiting for clients...\n")

    def update_loop():
        counter = 0
        try:
            while simulator.is_running:
                time.sleep(1)
                simulator.update_sensors()
                values = simulator.get_values()
                temp_node.set_value(values["temperature"])
                pressure_node.set_value(values["pressure"])
                speed_node.set_value(values["speed"])
                last_update.set_value(datetime.now(timezone.utc))
                counter += 1
                print(
                    f"[SERVER] Update #{counter}: "
                    f"T={values['temperature']:.1f} degC, "
                    f"P={values['pressure']:.1f} bar, "
                    f"S={values['speed']:.0f} RPM"
                )
        except Exception as exc:
            print(f"[SERVER] Update loop error: {exc}")
        finally:
            print("[SERVER] Update loop stopped")

    threading.Thread(target=update_loop, daemon=True).start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[SERVER] Shutdown requested")
    finally:
        simulator.is_running = False
        server.stop()
        print("[SERVER] Server stopped")


if __name__ == "__main__":
    main()
