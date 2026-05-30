import argparse
from pathlib import Path

from siome5_generic_exporter import export_siome5_with_mapping


def main():
    parser = argparse.ArgumentParser(
        description="Export any OPC UA server data into SIOME5 XML format"
    )

    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    xml_data = export_siome5_with_mapping(
        endpoint_url=args.endpoint,
        template_path=args.template,
        mapping_path=args.mapping,
    )

    output_path = Path(args.output)
    output_path.write_bytes(xml_data)

    print(f"SIOME5 export completed: {output_path}")


if __name__ == "__main__":
    main()