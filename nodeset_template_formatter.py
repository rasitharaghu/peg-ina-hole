from pathlib import Path
from html import escape
import re


def generate_nodeset_xml_from_nodes(nodes, template_path: str) -> bytes:
    """
    Current interface compatible:
        nodes = server.nodes.all()

    Behavior:
        - Reads sample XML template as plain text
        - Does NOT parse/rebuild XML
        - Replaces only existing value text inside <Value>...</Value>
        - Keeps template formatting/structure unchanged
    """

    template_text = Path(template_path).read_text(encoding="utf-8")

    runtime_values = _extract_runtime_values_from_nodes(nodes)

    updated_text = update_template_values_exact(
        template_text=template_text,
        runtime_values=runtime_values,
    )

    return updated_text.encode("utf-8")


def update_template_values_exact(template_text: str, runtime_values: dict) -> str:
    updated_text = template_text

    for field_name, value in runtime_values.items():
        updated_text = _replace_value_for_field(
            xml_text=updated_text,
            field_name=field_name,
            new_value=value,
        )

    return updated_text


def _replace_value_for_field(xml_text: str, field_name: str, new_value) -> str:
    """
    Finds:
        <UAVariable ... BrowseName="4:Temperature" ...>
            ...
            <Value>
                <uax:Double>0</uax:Double>
            </Value>
        </UAVariable>

    Replaces only:
        0 -> new_value
    """

    variable_pattern = re.compile(
        rf'(<UAVariable\b[^>]*BrowseName="(?:\d+:)?{re.escape(field_name)}"[^>]*>.*?</UAVariable>)',
        re.DOTALL,
    )

    match = variable_pattern.search(xml_text)

    if not match:
        return xml_text

    variable_block = match.group(1)

    updated_block = _replace_first_value_text_only(
        variable_block=variable_block,
        new_value=_format_xml_value(new_value),
    )

    return (
        xml_text[:match.start(1)]
        + updated_block
        + xml_text[match.end(1):]
    )


def _replace_first_value_text_only(variable_block: str, new_value: str) -> str:
    """
    Replaces only text inside the first uax value tag inside <Value>...</Value>.

    Example:
        <Value>
            <uax:Double>0</uax:Double>
        </Value>

    becomes:
        <Value>
            <uax:Double>42.5</uax:Double>
        </Value>
    """

    value_pattern = re.compile(
        r'(<Value>\s*.*?<uax:[A-Za-z0-9_]+\b[^>]*>)(.*?)(</uax:[A-Za-z0-9_]+>.*?</Value>)',
        re.DOTALL,
    )

    match = value_pattern.search(variable_block)

    if not match:
        return variable_block

    return (
        variable_block[:match.start(2)]
        + new_value
        + variable_block[match.end(2):]
    )


def _extract_runtime_values_from_nodes(nodes) -> dict:
    """
    Extracts values from current Django ORM node objects.

    Expected:
        node.label
        node.readings.first().value
    """

    values = {}

    for node in nodes:
        field_name = getattr(node, "label", None)

        if not field_name:
            field_name = getattr(node, "node_id", "")

        field_name = str(field_name).split(".")[-1].strip()

        if not field_name:
            continue

        try:
            latest_reading = node.readings.first()

            if latest_reading is None:
                continue

            if hasattr(latest_reading, "typed_value"):
                value = latest_reading.typed_value
            else:
                value = latest_reading.value

            values[field_name] = value

        except Exception:
            continue

    return values


def _format_xml_value(value) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return str(value).lower()

    return escape(str(value))