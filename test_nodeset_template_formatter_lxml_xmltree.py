from pathlib import Path

# Choose one:
from opcuainterface.nodeset_template_formatter_lxml import generate_nodeset_xml_from_nodes
# from opcuainterface.nodeset_template_formatter_xmltree import generate_nodeset_xml_from_nodes


class MockReading:
    def __init__(self, value):
        self.value = value
        self.typed_value = value


class MockReadings:
    def __init__(self, value):
        self._value = value

    def first(self):
        return MockReading(self._value)


class MockNode:
    def __init__(self, label, value):
        self.label = label
        self.readings = MockReadings(value)


def main():
    template_path = "AtlasCopco-Tools.Nodeset2_ToBeWantedSample.xml"

    mock_nodes = [
        MockNode("Temperature", 42.5),
        MockNode("WorkOrder", "WO_123"),
        MockNode("TaskIdentifier", "TASK_001"),
        MockNode("MaxTorque", 150.0),
        MockNode("ActStatus", "Running"),
    ]

    xml_bytes = generate_nodeset_xml_from_nodes(
        nodes=mock_nodes,
        template_path=template_path,
    )

    output_path = Path("Generated_Test_NodeSet.xml")
    output_path.write_bytes(xml_bytes)

    print("Generated XML:", output_path.resolve())


if __name__ == "__main__":
    main()