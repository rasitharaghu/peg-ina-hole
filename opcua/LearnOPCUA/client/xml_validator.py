from lxml import etree
import os
from datetime import datetime


def validate_xml_with_xsd(xml_path, xsd_path):
    """Validate an XML file against a provided XSD.

    Returns:
        (bool, str): (is_valid, error_message). If valid, error_message is empty.
    """
    if not os.path.exists(xml_path):
        return False, f"XML file not found: {xml_path}"
    if not os.path.exists(xsd_path):
        return False, f"XSD file not found: {xsd_path}"

    try:
        with open(xsd_path, 'rb') as f:
            xsd_doc = etree.parse(f)
        schema = etree.XMLSchema(xsd_doc)
    except (etree.XMLSchemaParseError, etree.XMLSyntaxError) as e:
        return False, f"Failed to parse XSD: {e}"
    except Exception as e:
        return False, f"Unexpected error loading XSD: {e}"

    try:
        parser = etree.XMLParser(remove_blank_text=True)
        xml_doc = etree.parse(xml_path, parser)
    except etree.XMLSyntaxError as e:
        return False, f"Failed to parse XML: {e}"
    except Exception as e:
        return False, f"Unexpected error parsing XML: {e}"

    try:
        schema.assertValid(xml_doc)
        return True, ""
    except etree.DocumentInvalid as e:
        # Collect detailed log
        error_log = schema.error_log
        messages = [str(entry) for entry in error_log]
        return False, "\n".join(messages)
    except Exception as e:
        return False, f"Unexpected validation error: {e}"


def log_validation_error(log_path, xml_path, error_text):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] Validation failed for: {xml_path}\n")
        f.write(error_text + "\n")
        f.write("-" * 60 + "\n")


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('xml', help='Path to XML file')
    p.add_argument('xsd', help='Path to XSD file')
    args = p.parse_args()

    valid, err = validate_xml_with_xsd(args.xml, args.xsd)
    if valid:
        print('VALID')
        exit(0)
    else:
        print('INVALID')
        print(err)
        exit(2)
