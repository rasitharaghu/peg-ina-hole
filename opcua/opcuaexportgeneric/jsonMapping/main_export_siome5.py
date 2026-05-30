import argparse
from pathlib import Path

from siome5_generic_exporter import (
    export_siome5_with_mapping,
    auto_create_mapping_from_server,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export OPC UA server values into SIOME5 XML format"
    )

    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--mapping",
        required=False,
        help="Optional JSON mapping file. If not given, auto-mapping is used.",
    )

    parser.add_argument(
        "--save-auto-mapping",
        required=False,
        help="Optional path to save generated auto-mapping JSON.",
    )

    args = parser.parse_args()

    if args.mapping:
        mapping_source = args.mapping
        print(f"Using mapping file: {args.mapping}")
    else:
        print("No mapping file provided. Auto-mapping by BrowseName similarity...")
        mapping_source = auto_create_mapping_from_server(
            endpoint_url=args.endpoint,
            template_path=args.template,
            save_path=args.save_auto_mapping,
        )

    xml_data = export_siome5_with_mapping(
        endpoint_url=args.endpoint,
        template_path=args.template,
        mapping=mapping_source,
    )

    Path(args.output).write_bytes(xml_data)

    print(f"SIOME5 XML exported successfully: {args.output}")


if __name__ == "__main__":
    main()