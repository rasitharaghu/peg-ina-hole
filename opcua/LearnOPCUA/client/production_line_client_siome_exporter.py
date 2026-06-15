"""
Production Line OPC UA Client + SIOME-compatible UANodeSet XML exporter.

What this code does:
    - Connects to an OPC UA server.
    - Browses the live OPC UA address space from Objects.
    - Exports UAObject, UAVariable and UAMethod nodes into UANodeSet-style XML.
    - Builds Aliases dynamically from the server type/reference trees.
    - Uses Opc.Ua.Types.xsd, when available, to know valid uax:* value elements.
    - Automatically creates timestamped XML files in the output folder.

Install:
    pip install opcua lxml

Run:
    python production_line_client_siome_exporter.py
"""

import os
import re
import time
import hashlib
from datetime import datetime, timezone
from xml.etree.ElementTree import Element, SubElement, ElementTree, register_namespace, QName
import xml.etree.ElementTree as ET

from opcua import Client, ua
from pathlib import Path


UANODESET_NS = "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd"
UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"
SI_NS = "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XSD_NS = "http://www.w3.org/2001/XMLSchema"


class ProductionLineClient:
    """OPC UA client that can browse and export a SIOME-compatible NodeSet XML."""

    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.client = Client(endpoint)

        self.alias_map = {}
        self.reverse_alias_map = {}
        self.uax_direct_value_types = set()
        self.uax_complex_type_fields = {}

        # No XML filename is required from main(). File is auto-created in output_dir.
        # self.default_types_xsd = os.path.join(os.path.dirname(__file__), "Opc.Ua.Types.xsd")
        # xml_dir = os.path.dirname(__file__)

        # self.default_types_xsd = os.path.join(xml_dir, "Opc.Ua.Types.xsd")
        script_dir = Path(__file__).resolve().parent

        xml_dir = script_dir.parent / "xml"

        self.default_types_xsd = xml_dir / "Opc.Ua.Types.xsd"

        self.model_xsd_paths = [
            os.path.join(xml_dir, "Opc.Ua.Types.xsd"),
            os.path.join(xml_dir, "Opc.Ua.Ijt.Base.Types.xsd"),
            os.path.join(xml_dir, "Opc.Ua.Ijt.Tightening.Types.xsd"),
            os.path.join(xml_dir, "Opc.Ua.Machinery.Result.Types.xsd"),
        ]

        self.server_datatype_fields = {}
        self.export_namespace_uris = {
            "http://learning.opcua.production"
        }

        self.complex_type_fallback_fields = {
            "EUInformation": ["NamespaceUri", "UnitId", "DisplayName", "Description"],

            "StepTraceDataType": [
                "EncodingMask",
                "NumberOfTracePoints",
                "SamplingInterval",
                "StartTimeOffset",
            ],

            "TraceContentDataType": [
                "EncodingMask",
            ],

            "ResultDataType": [
                "EncodingMask",
            ],

            "TighteningTraceDataType": [
                "EncodingMask",
            ],
        }
        self.enable_hash_extensions = False
        # self.server_datatype_fields = {}

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(self):
        try:
            self.client.connect()
            print("[CLIENT] Connected successfully!")
            return True
        except Exception as e:
            print(f"[CLIENT] Connection failed: {e}")
            return False

    def disconnect(self):
        try:
            self.client.disconnect()
            print("[CLIENT] Disconnected successfully")
        except Exception as e:
            print(f"[CLIENT] Disconnect error: {e}")

    # -------------------------------------------------------------------------
    # Browse dump
    # -------------------------------------------------------------------------

    def browse_namespace(self, node=None, indent=0, output_file=None):
        """Recursively browse and print the namespace tree."""
        if node is None:
            node = self.client.get_root_node()

        try:
            display = node.get_display_name().Text
            output_text = "  " * indent + f"├─ {display} (ID: {node.nodeid.to_string()})"

            if output_file:
                output_file.write(output_text + "\n")
            else:
                print(output_text)

            for child in node.get_children():
                self.browse_namespace(child, indent + 1, output_file)

        except Exception as e:
            error_text = f"[CLIENT] Browse error: {e}"
            if output_file:
                output_file.write(error_text + "\n")
            else:
                print(error_text)

    def dump_browse_namespace_to_file(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        file_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        file_path = os.path.join(output_dir, f"browse_namespace_{file_timestamp}.txt")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"[CLIENT] Browse Namespace - Timestamp: {datetime.utcnow().isoformat()}Z\n")
                f.write("─" * 50 + "\n")
                self.browse_namespace(output_file=f)
                f.write("─" * 50 + "\n")
            print(f"[CLIENT] Wrote browse namespace dump: {file_path}")
        except Exception as e:
            print(f"[CLIENT] Browse namespace dump error: {e}")

    # -------------------------------------------------------------------------
    # XML root / namespace setup
    # -------------------------------------------------------------------------

    # def create_uanodeset_root(self):
    #     """Create UANodeSet root following UANodeSet.xsd style."""
    #     register_namespace("", UANODESET_NS)
    #     register_namespace("xsi", XSI_NS)
    #     register_namespace("uax", UAX_NS)
    #     register_namespace("si", SI_NS)
    #     register_namespace("xsd", XSD_NS)

    #     return Element("UANodeSet", {
    #         "LastModified": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
    #         "xmlns": UANODESET_NS,
    #         "xmlns:xsi": XSI_NS,
    #         "xmlns:uax": UAX_NS,
    #         "xmlns:si": SI_NS,
    #         "xmlns:xsd": XSD_NS,
    #     })

    def create_uanodeset_root(self):
        return Element("UANodeSet", {
            "LastModified": datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def make_symbolic_name(self, text):
        if text is None:
            return ""
        text = str(text)
        text = re.sub(r"[^0-9A-Za-z_]", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            return ""
        if text[0].isdigit():
            text = "S" + text
        return text

    def qname_to_string(self, qname):
        try:
            ns_idx = qname.NamespaceIndex
            name = qname.Name
            if ns_idx == 0:
                return name
            return f"{ns_idx}:{name}"
        except Exception:
            return str(qname)

    # def find_child_by_browse_name(self, parent, browse_name):
    #     try:
    #         for child in parent.get_children():
    #             child_browse_name = child.get_browse_name()
    #             if hasattr(child_browse_name, "Name") and child_browse_name.Name == browse_name:
    #                 return child
    #     except Exception:
    #         pass
    #     return None

    def get_child_value(self, parent, child_name):
        child = self.find_child_by_browse_name(parent, child_name)
        if child is None:
            return ""

        try:
            value = child.get_value()
            return self.format_value_for_xml(value)
        except Exception:
            return ""

    def format_value_for_xml(self, value):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat().replace("+00:00", "Z")

        if isinstance(value, bool):
            return str(value).lower()

        if hasattr(value, "Text") and hasattr(value, "Locale"):
            return str(value.Text or "")

        return str(value)

    # -------------------------------------------------------------------------
    # NamespaceUris / Models / Aliases / Extensions
    # -------------------------------------------------------------------------

    def add_namespace_uris(self, root):
        """Add NamespaceUris from the server NamespaceArray."""
        namespace_uris_el = SubElement(root, "NamespaceUris")

        try:
            namespace_uris = self.client.get_namespace_array()
        except Exception:
            namespace_uris = []

        # SIOME sample normally omits namespace 0, but many tools tolerate both.
        # Here we skip namespace 0 because the default UANodeSet schema already implies OPC UA base.
        for uri in namespace_uris[1:]:
            SubElement(namespace_uris_el, "Uri").text = str(uri)

    def add_models(self, root):
        """
        Try to add Models from Objects.Server.Namespaces.
        If the server does not expose NamespaceMetadataType instances, this section remains empty.
        """
        models_el = SubElement(root, "Models")

        try:
            objects_node = self.client.get_objects_node()
            server_node = self.find_child_by_browse_name(objects_node, "Server")
            if server_node is None:
                return

            namespaces_node = self.find_child_by_browse_name(server_node, "Namespaces")
            if namespaces_node is None:
                return

            for namespace_item in namespaces_node.get_children():
                namespace_uri = self.get_child_value(namespace_item, "NamespaceUri")
                namespace_pub_date = self.get_child_value(namespace_item, "NamespacePublicationDate")
                namespace_version = self.get_child_value(namespace_item, "NamespaceVersion")

                if not namespace_uri:
                    continue

                attrs = {"ModelUri": namespace_uri}
                if namespace_pub_date:
                    attrs["PublicationDate"] = namespace_pub_date
                if namespace_version:
                    attrs["Version"] = namespace_version

                model_el = SubElement(models_el, "Model", attrs)

                # RequiredModel may be exposed under NamespaceMetadata.RequiredModels.
                required_models = self.find_child_by_browse_name(namespace_item, "RequiredModels")
                if required_models is not None:
                    try:
                        for req in required_models.get_children():
                            req_uri = self.get_child_value(req, "ModelUri") or self.get_child_value(req, "NamespaceUri")
                            req_pub = self.get_child_value(req, "PublicationDate")
                            req_ver = self.get_child_value(req, "Version")
                            if req_uri:
                                req_attrs = {"ModelUri": req_uri}
                                if req_pub:
                                    req_attrs["PublicationDate"] = req_pub
                                if req_ver:
                                    req_attrs["Version"] = req_ver
                                SubElement(model_el, "RequiredModel", req_attrs)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[CLIENT] Model generation warning: {e}")

    # def add_aliases(self, root):
    #     """
    #     Build aliases dynamically from live server type/reference trees.
    #     No hardcoded Boolean/Double/HasComponent aliases.
    #     """
    #     aliases_el = SubElement(root, "Aliases")
    #     alias_map = {}

    #     type_root_ids = [
    #         ua.ObjectIds.DataTypes,
    #         ua.ObjectIds.ReferenceTypes,
    #         ua.ObjectIds.ObjectTypes,
    #         ua.ObjectIds.VariableTypes,
    #     ]

    #     queue = []
    #     for root_id in type_root_ids:
    #         try:
    #             queue.append(self.client.get_node(ua.NodeId(root_id, 0)))
    #         except Exception:
    #             pass

    #     visited = set()

    #     while queue:
    #         current_node = queue.pop(0)
    #         nodeid_str = current_node.nodeid.to_string()
    #         if nodeid_str in visited:
    #             continue
    #         visited.add(nodeid_str)

    #         try:
    #             browse_name = current_node.get_browse_name().Name
    #             alias_name = self.make_symbolic_name(browse_name)
    #             if alias_name and alias_name not in alias_map:
    #                 alias_map[alias_name] = nodeid_str
    #         except Exception:
    #             pass

    #         try:
    #             for child in current_node.get_children():
    #                 queue.append(child)
    #         except Exception:
    #             pass

    #     for alias_name, nodeid in sorted(alias_map.items()):
    #         alias_el = SubElement(aliases_el, "Alias", {"Alias": alias_name})
    #         alias_el.text = nodeid

    #     self.alias_map = alias_map
    #     self.reverse_alias_map = {nodeid: alias for alias, nodeid in alias_map.items()}

    def find_child_by_browse_name(self, parent, browse_name):
        try:
            for child in parent.get_children():
                bn = child.get_browse_name()
                if bn.Name == browse_name:
                    return child
        except Exception:
            pass
        return None


    # def add_aliases(self, root):
    #     """Build aliases dynamically by browsing Root.Types from the live server."""

    #     aliases_el = SubElement(root, "Aliases")
    #     alias_map = {}

    #     try:
    #         root_node = self.client.get_root_node()

    #         types_node = self.find_child_by_browse_name(root_node, "Types")
    #         if types_node is None:
    #             print("[CLIENT] Warning: Root.Types not found; aliases will be empty")
    #             self.alias_map = {}
    #             self.reverse_alias_map = {}
    #             return

    #         type_folder_names = [
    #             "DataTypes",
    #             "ReferenceTypes",
    #             "ObjectTypes",
    #             "VariableTypes",
    #         ]

    #         queue = []

    #         for folder_name in type_folder_names:
    #             folder_node = self.find_child_by_browse_name(types_node, folder_name)
    #             if folder_node is not None:
    #                 queue.append(folder_node)

    #         visited = set()

    #         while queue:
    #             node = queue.pop(0)
    #             nodeid_str = node.nodeid.to_string()

    #             if nodeid_str in visited:
    #                 continue

    #             visited.add(nodeid_str)

    #             try:
    #                 browse_name = node.get_browse_name().Name
    #             except Exception:
    #                 browse_name = ""

    #             if browse_name:
    #                 safe_alias = self.make_symbolic_name(browse_name)

    #                 if safe_alias and safe_alias not in alias_map:
    #                     alias_map[safe_alias] = nodeid_str

    #             try:
    #                 for child in node.get_children():
    #                     queue.append(child)
    #             except Exception:
    #                 pass

    #     except Exception as e:
    #         print(f"[CLIENT] Dynamic alias discovery error: {e}")

    #     for alias_name, nodeid in sorted(alias_map.items()):
    #         alias_el = SubElement(aliases_el, "Alias", {"Alias": alias_name})
    #         alias_el.text = nodeid

    #     self.alias_map = alias_map
    #     self.reverse_alias_map = {
    #         nodeid: alias for alias, nodeid in alias_map.items()
    #     }

    #     print(f"[CLIENT] Created {len(alias_map)} aliases dynamically")

    def get_types_root_folders(self):
        """
        Dynamically find:
            Root -> Types -> DataTypes
            Root -> Types -> ReferenceTypes
            Root -> Types -> ObjectTypes
            Root -> Types -> VariableTypes

        No hardcoded i=24, i=31, i=58, i=62.
        """
        folders = []

        try:
            root_node = self.client.get_root_node()
            types_node = self.find_child_by_browse_name(root_node, "Types")

            if types_node is None:
                print("[CLIENT] Warning: Root.Types not found")
                return folders

            for folder_name in ["DataTypes", "ReferenceTypes", "ObjectTypes", "VariableTypes"]:
                folder = self.find_child_by_browse_name(types_node, folder_name)
                if folder is not None:
                    folders.append(folder)
                else:
                    print(f"[CLIENT] Warning: Root.Types.{folder_name} not found")

        except Exception as e:
            print(f"[CLIENT] Type root discovery error: {e}")

        return folders


    def add_aliases(self, root):
        """Build aliases dynamically from Root.Types address space."""

        aliases_el = SubElement(root, "Aliases")
        alias_map = {}

        queue = self.get_types_root_folders()
        visited = set()

        while queue:
            node = queue.pop(0)

            try:
                nodeid_str = node.nodeid.to_string()
            except Exception:
                continue

            if nodeid_str in visited:
                continue
            visited.add(nodeid_str)

            try:
                browse_name = node.get_browse_name().Name
                alias_name = self.make_symbolic_name(browse_name)

                if alias_name and alias_name not in alias_map:
                    alias_map[alias_name] = nodeid_str

            except Exception:
                pass

            try:
                for child in node.get_children():
                    queue.append(child)
            except Exception:
                pass

        for alias_name, nodeid in sorted(alias_map.items()):
            alias_el = SubElement(aliases_el, "Alias", {"Alias": alias_name})
            alias_el.text = nodeid

        self.alias_map = alias_map
        self.reverse_alias_map = {
            nodeid: alias for alias, nodeid in alias_map.items()
        }

        print(f"[CLIENT] Created {len(alias_map)} aliases dynamically from Root.Types")

    def generate_extension_hashes(self):
        structural_buffer = []
        contextual_buffer = []

        try:
            objects_node = self.client.get_objects_node()
            queue = [objects_node]
            visited = set()

            while queue:
                node = queue.pop(0)
                nodeid = node.nodeid.to_string()
                if nodeid in visited:
                    continue
                visited.add(nodeid)

                try:
                    browse_name = self.qname_to_string(node.get_browse_name())
                    display_name = node.get_display_name().Text
                    structural_buffer.append(f"{nodeid}|{browse_name}")
                    contextual_buffer.append(f"{nodeid}|{browse_name}|{display_name}")

                    try:
                        value = node.get_value()
                        contextual_buffer.append(str(value))
                    except Exception:
                        pass

                    for child in node.get_children():
                        queue.append(child)
                except Exception:
                    continue

        except Exception as e:
            print(f"[CLIENT] Hash generation warning: {e}")

        h1 = hashlib.md5("".join(structural_buffer).encode("utf-8")).hexdigest()
        h2 = hashlib.md5("".join(contextual_buffer).encode("utf-8")).hexdigest()
        return h1, h2

    def add_extensions(self, root, hash_ns1, hash_si):
        extensions_el = SubElement(root, "Extensions")

        ext1 = SubElement(extensions_el, "Extension")
        SubElement(ext1, QName(SI_NS, "Generator"), {
            "Product": "SiOME",
            "Edition": "Sinumerik",
            "Version": "2.8.5-installer",
        })

        ext2 = SubElement(extensions_el, "Extension")
        SubElement(ext2, QName(SI_NS, "GeneratorExtension"), {
            "Hash": hash_si,
        })

        # Optional custom exporter metadata.
        ext3 = SubElement(extensions_el, "Extension")
        SubElement(ext3, QName(SI_NS, "Generator"), {
            "Product": "Custom OPC UA NodeSet Exporter",
            "Edition": "Generic",
            "Version": "1.0",
        })

        ext4 = SubElement(extensions_el, "Extension")
        SubElement(ext4, QName(SI_NS, "GeneratorExtension"), {
            "Hash": hash_ns1,
        })

    # -------------------------------------------------------------------------
    # Opc.Ua.Types.xsd support
    # -------------------------------------------------------------------------

    def load_uax_direct_value_types(self, types_xsd_path=None):
        """
        Load valid uax:* element names from Opc.Ua.Types.xsd.
        If the file is missing, use a safe built-in fallback for common scalar types.
        """
        fallback = {
            "Boolean", "SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32",
            "Int64", "UInt64", "Float", "Double", "String", "DateTime",
            "Guid", "ByteString", "XmlElement", "NodeId", "ExpandedNodeId",
            "StatusCode", "QualifiedName", "LocalizedText", "ExtensionObject",
            "ListOfBoolean", "ListOfByte", "ListOfInt16", "ListOfUInt16",
            "ListOfInt32", "ListOfUInt32", "ListOfInt64", "ListOfUInt64",
            "ListOfFloat", "ListOfDouble", "ListOfString", "ListOfDateTime",
            "ListOfLocalizedText", "EUInformation", "EnumValueType",
        }

        self.uax_direct_value_types = set(fallback)

        if not types_xsd_path:
            types_xsd_path = self.default_types_xsd

        if not os.path.exists(types_xsd_path):
            print(
                f"[CLIENT] Opc.Ua.Types.xsd not found: {types_xsd_path}. "
                "Using fallback scalar value types only. "
                "Complex/custom structures may not export correctly."
            )
            return

        try:
            xsd_ns = "{http://www.w3.org/2001/XMLSchema}"
            tree = ET.parse(types_xsd_path)
            xsd_root = tree.getroot()

            value_types = set()
            for element in xsd_root.findall(f".//{xsd_ns}element"):
                name = element.attrib.get("name")
                if name:
                    value_types.add(name)

            if value_types:
                self.uax_direct_value_types = value_types
                print(f"[CLIENT] Loaded {len(value_types)} uax value element names from {types_xsd_path}")

        except Exception as e:
            print(f"[CLIENT] Could not parse Opc.Ua.Types.xsd, using fallback: {e}")

    # -------------------------------------------------------------------------
    # Value encoding
    # -------------------------------------------------------------------------

    NUMERIC_UAX_TYPES = {
        "SByte", "Byte", "Int16", "UInt16", "Int32", "UInt32", "Int64", "UInt64",
        "Integer", "UInteger",
    }

    FLOAT_UAX_TYPES = {"Float", "Double", "Duration", "Number"}


    # def load_uax_complex_type_fields(self, types_xsd_path=None):
    #     """
    #     Load complex OPC UA XML type fields from Opc.Ua.Types.xsd.

    #     Example:
    #         EUInformation -> NamespaceUri, UnitId, DisplayName, Description
    #     """

    #     self.uax_complex_type_fields = {}

    #     if not types_xsd_path:
    #         types_xsd_path = self.default_types_xsd

    #     if not os.path.exists(types_xsd_path):
    #         print(f"[CLIENT] Opc.Ua.Types.xsd not found for complex types: {types_xsd_path}")
    #         return

    #     try:
    #         xsd_ns = "{http://www.w3.org/2001/XMLSchema}"
    #         tree = ET.parse(types_xsd_path)
    #         xsd_root = tree.getroot()

    #         for complex_type in xsd_root.findall(f".//{xsd_ns}complexType"):
    #             type_name = complex_type.attrib.get("name")
    #             if not type_name:
    #                 continue

    #             fields = []
    #             for element in complex_type.findall(f".//{xsd_ns}element"):
    #                 field_name = element.attrib.get("name")
    #                 if field_name:
    #                     fields.append(field_name)

    #             if fields:
    #                 self.uax_complex_type_fields[type_name] = fields

    #         print(f"[CLIENT] Loaded {len(self.uax_complex_type_fields)} complex uax type definitions")

    #     except Exception as e:
    #         print(f"[CLIENT] Could not parse complex types from Opc.Ua.Types.xsd: {e}")

    def load_uax_complex_type_fields(self, types_xsd_path=None):
        if not types_xsd_path:
            types_xsd_path = self.default_types_xsd

        if not os.path.exists(types_xsd_path):
            print(f"[CLIENT] Types XSD not found, skipping: {types_xsd_path}")
            return

        try:
            xsd_ns = "{http://www.w3.org/2001/XMLSchema}"
            tree = ET.parse(types_xsd_path)
            xsd_root = tree.getroot()

            loaded_count = 0

            for complex_type in xsd_root.findall(f".//{xsd_ns}complexType"):
                type_name = complex_type.attrib.get("name")
                if not type_name:
                    continue

                fields = []
                for element in complex_type.findall(f".//{xsd_ns}element"):
                    field_name = element.attrib.get("name")
                    if field_name:
                        fields.append(field_name)

                if fields:
                    self.uax_complex_type_fields[type_name] = fields
                    loaded_count += 1

            print(f"[CLIENT] Loaded {loaded_count} complex types from {types_xsd_path}")

        except Exception as e:
            print(f"[CLIENT] Could not parse complex types from {types_xsd_path}: {e}")


    def format_value_for_uax(self, data_type_name, value):
        if value is None:
            return None

        if data_type_name == "DateTime" or data_type_name == "UtcTime":
            return self.format_value_for_xml(value)

        if data_type_name == "Boolean":
            if isinstance(value, str):
                return value.strip().lower()
            return str(bool(value)).lower()

        if data_type_name in self.FLOAT_UAX_TYPES:
            try:
                return str(float(value))
            except Exception:
                return None

        if data_type_name in self.NUMERIC_UAX_TYPES:
            try:
                return str(int(value))
            except Exception:
                return None

        # if data_type_name == "LocalizedText":
        #     if hasattr(value, "Text"):
        #         return str(value.Text or "")
        #     return str(value)

        return self.format_value_for_xml(value)

    def add_localized_text_value(self, value_el, value):
        lt_el = SubElement(value_el, QName(UAX_NS, "LocalizedText"))
        text_el = SubElement(lt_el, QName(UAX_NS, "Text"))
        if hasattr(value, "Text"):
            text_el.text = value.Text or ""
        else:
            text_el.text = str(value)

    def add_list_value(self, variable_el, data_type_name, value):
        if not isinstance(value, (list, tuple)):
            return False

        list_tag = f"ListOf{data_type_name}"
        if list_tag not in self.uax_direct_value_types:
            return False

        value_el = SubElement(variable_el, "Value")
        list_el = SubElement(value_el, QName(UAX_NS, list_tag))

        for item in value:
            if data_type_name == "LocalizedText":
                self.add_localized_text_value(list_el, item)
            else:
                item_el = SubElement(list_el, QName(UAX_NS, data_type_name))
                formatted = self.format_value_for_uax(data_type_name, item)
                if formatted is not None:
                    item_el.text = formatted

        return True
    

    def parse_structured_value(self, value):
        """Convert structured OPC UA values / JSON strings / dicts into a dict."""

        if value is None:
            return None

        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    import json
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None

        result = {}

        for attr in dir(value):
            if attr.startswith("_"):
                continue
            if attr in {"ua_types", "ua_switches"}:
                continue

            try:
                attr_value = getattr(value, attr)
            except Exception:
                continue

            if callable(attr_value):
                continue

            if attr_value is not None:
                result[attr] = attr_value

        return result if result else None


    def add_generic_field_value(self, parent_el, field_value):
        """Write one field value inside an ExtensionObject body."""

        if field_value is None:
            return False

        # LocalizedText object
        if hasattr(field_value, "Text"):
            if hasattr(field_value, "Locale") and field_value.Locale:
                locale_el = SubElement(parent_el, QName(UAX_NS, "Locale"))
                locale_el.text = str(field_value.Locale)

            text_el = SubElement(parent_el, QName(UAX_NS, "Text"))
            text_el.text = str(field_value.Text or "")
            return True

        # LocalizedText-like dict
        if isinstance(field_value, dict):
            if "Text" in field_value or "Locale" in field_value:
                if field_value.get("Locale"):
                    locale_el = SubElement(parent_el, QName(UAX_NS, "Locale"))
                    locale_el.text = str(field_value.get("Locale"))

                text_el = SubElement(parent_el, QName(UAX_NS, "Text"))
                text_el.text = str(field_value.get("Text", ""))
                return True

            for key, val in field_value.items():
                if key.startswith("__"):
                    continue
                child = SubElement(parent_el, QName(UAX_NS, key))
                self.add_generic_field_value(child, val)
            return True

        # List / array
        if isinstance(field_value, (list, tuple)):
            for item in field_value:
                item_el = SubElement(parent_el, QName(UAX_NS, "Element"))
                self.add_generic_field_value(item_el, item)
            return True

        parent_el.text = self.format_value_for_xml(field_value)
        return True


    def add_extension_object_value_element(self, variable_el, data_type_name, value):
        """
        Generic ExtensionObject writer.

        Uses:
        - datatype name from server
        - binary encoding id from get_binary_encoding_id()
        - fields from Opc.Ua.Types.xsd when available
        - JSON/dict/object fields otherwise
        """

        structured = self.parse_structured_value(value)
        if not structured:
            return False

        actual_type = structured.get("__DataType", data_type_name)

        encoding_id = (
            structured.get("__TypeId")
            or structured.get("__BinaryEncodingId")
            or self.get_binary_encoding_id(actual_type)
        )

        if not encoding_id:
            return False

        # field_names = self.uax_complex_type_fields.get(actual_type)

        # if not field_names:
        #     field_names = [
        #         key for key in structured.keys()
        #         if not key.startswith("__")
        #     ]

        field_names = self.get_datatype_definition_fields(actual_type)

        if not field_names:
            field_names = self.uax_complex_type_fields.get(actual_type)

        if not field_names:
            field_names = [
                key for key in structured.keys()
                if not key.startswith("__")
            ]

        value_el = SubElement(variable_el, "Value")

        ext_obj_el = SubElement(value_el, QName(UAX_NS, "ExtensionObject"))

        type_id_el = SubElement(ext_obj_el, QName(UAX_NS, "TypeId"))
        identifier_el = SubElement(type_id_el, QName(UAX_NS, "Identifier"))
        identifier_el.text = str(encoding_id)

        body_el = SubElement(ext_obj_el, QName(UAX_NS, "Body"))
        body_type_el = SubElement(body_el, QName(UAX_NS, actual_type))

        written = False

        for field_name in field_names:
            if field_name.startswith("__"):
                continue

            if field_name not in structured:
                continue

            field_value = structured.get(field_name)
            if field_value is None:
                continue

            field_el = SubElement(body_type_el, QName(UAX_NS, field_name))

            if self.add_generic_field_value(field_el, field_value):
                written = True
            else:
                body_type_el.remove(field_el)

        if not written:
            variable_el.remove(value_el)
            return False

        return True

    def add_typed_value_element(self, variable_el, data_type_name, value):
        """Add <Value><uax:Type>...</uax:Type></Value> based on OPC UA XML encoding."""
        if value is None or not data_type_name:
            return

        # Arrays/lists
        if isinstance(value, (list, tuple)):
            self.add_list_value(variable_el, data_type_name, value)
            return

        # LocalizedText needs nested Text element.
        if (
            data_type_name in self.uax_complex_type_fields
            and data_type_name not in self.alias_map
        ):
            value_el = SubElement(variable_el, "Value")
            direct_el = SubElement(value_el, QName(UAX_NS, data_type_name))

            structured = self.parse_structured_value(value)

            if structured:
                field_names = self.uax_complex_type_fields.get(data_type_name, [])
                for field_name in field_names:
                    if field_name not in structured:
                        continue

                    field_value = structured.get(field_name)
                    if field_value is None:
                        continue

                    field_el = SubElement(direct_el, QName(UAX_NS, field_name))
                    self.add_generic_field_value(field_el, field_value)
            else:
                self.add_generic_field_value(direct_el, value)

            return
        
        if data_type_name == "LocalizedText":
            value_el = SubElement(variable_el, "Value")
            lt_el = SubElement(value_el, QName(UAX_NS, "LocalizedText"))

            if hasattr(value, "Locale") and value.Locale:
                locale_el = SubElement(lt_el, QName(UAX_NS, "Locale"))
                locale_el.text = str(value.Locale)

            text_el = SubElement(lt_el, QName(UAX_NS, "Text"))

            if hasattr(value, "Text"):
                text_el.text = value.Text or ""
            else:
                text_el.text = str(value)

            return

        # ExtensionObject / custom structures are intentionally not guessed.
        # They need field definitions from the structure DataType or a decoded ExtensionObject.
        # if data_type_name in {"ExtensionObject", "EUInformation", "EnumValueType"}:
        #     # Minimal support for string/dict is possible, but avoid invalid structure.
        #     return

        # if data_type_name in self.alias_map:
        #     if self.add_extension_object_value_element(variable_el, data_type_name, value):
        #         return
            
        if (
            data_type_name in self.alias_map
            or data_type_name in self.uax_complex_type_fields
        ):
            if self.add_extension_object_value_element(
                variable_el,
                data_type_name,
                value
            ):
                return

        if data_type_name not in self.uax_direct_value_types:
            return

        formatted = self.format_value_for_uax(data_type_name, value)
        if formatted is None:
            return

        value_el = SubElement(variable_el, "Value")
        child = SubElement(value_el, QName(UAX_NS, data_type_name))
        child.text = formatted

    # -------------------------------------------------------------------------
    # Node XML builders
    # -------------------------------------------------------------------------

    def build_display_name_element(self, element, node):
        display_name = SubElement(element, "DisplayName")
        try:
            display_name.text = node.get_display_name().Text
        except Exception:
            display_name.text = str(node)

    def build_description_element(self, element, node):
        try:
            desc = node.get_description()
            if desc and desc.Text:
                desc_el = SubElement(element, "Description")
                desc_el.text = desc.Text
        except Exception:
            pass

    def build_references_element(self, element, node):
        """
        Export references.

        To avoid duplicating child links, we keep:
            - inverse parent references, e.g. HasComponent IsForward=false
            - forward HasTypeDefinition / HasInterface / HasAddIn
        """
        references_el = SubElement(element, "References")

        allowed_forward_refs = {
            "HasTypeDefinition",
            "HasInterface",
            "HasAddIn",
        }

        try:
            references = node.get_references(
                refs=ua.ObjectIds.References,
                direction=ua.BrowseDirection.Both,
                includesubtypes=True,
            )

            for ref in references:
                ref_type_nodeid = ref.ReferenceTypeId.to_string()
                ref_type = self.reverse_alias_map.get(ref_type_nodeid, ref_type_nodeid)

                # Skip forward child references; parent-child is already represented by child inverse references.
                if ref.IsForward and ref_type not in allowed_forward_refs:
                    continue

                ref_attribs = {"ReferenceType": ref_type}
                if ref.IsForward is False:
                    ref_attribs["IsForward"] = "false"

                reference_el = SubElement(references_el, "Reference", ref_attribs)
                reference_el.text = ref.NodeId.to_string() if hasattr(ref.NodeId, "to_string") else str(ref.NodeId)

        except Exception:
            pass

    # def get_variable_datatype_alias(self, node):
    #     try:
    #         data_type_nodeid = node.get_data_type()
    #         data_type_nodeid_str = data_type_nodeid.to_string()
    #         return self.reverse_alias_map.get(data_type_nodeid_str, data_type_nodeid_str)
    #     except Exception:
    #         return ""

    def get_variable_datatype_alias(self, node):
        try:
            data_type_nodeid = node.get_data_type()
            data_type_nodeid_str = data_type_nodeid.to_string()

            alias = self.reverse_alias_map.get(data_type_nodeid_str)
            if alias:
                return alias

            # Fallback: read BrowseName directly from the DataType node.
            try:
                dt_node = self.client.get_node(data_type_nodeid)
                return self.make_symbolic_name(dt_node.get_browse_name().Name)
            except Exception:
                pass

            return data_type_nodeid_str

        except Exception:
            return ""

    def get_access_level_int(self, node, attr_id):
        try:
            dv = node.get_attribute(attr_id)
            val = dv.Value.Value
            if isinstance(val, set):
                # python-opcua may return AccessLevel flags as a set.
                result = 0
                for item in val:
                    try:
                        result |= int(item.value)
                    except Exception:
                        pass
                return result
            return int(val)
        except Exception:
            return None

    def export_node(self, node, parent_nodeid, root, visited):
        try:
            nodeid_str = node.nodeid.to_string()
        except Exception:
            return

        if nodeid_str in visited:
            return
        visited.add(nodeid_str)

        try:
            node_class = node.get_node_class()
            browse_name = node.get_browse_name()
        except Exception:
            return

        if node_class == ua.NodeClass.Object:
            attrs = {
                "NodeId": nodeid_str,
                "BrowseName": self.qname_to_string(browse_name),
            }
            if parent_nodeid:
                attrs["ParentNodeId"] = parent_nodeid

            symbolic = self.make_symbolic_name(browse_name.Name if hasattr(browse_name, "Name") else str(browse_name))
            if symbolic:
                attrs["SymbolicName"] = symbolic

            element = SubElement(root, "UAObject", attrs)

        elif node_class == ua.NodeClass.Variable:
            data_type = self.get_variable_datatype_alias(node)

            attrs = {
                "DataType": data_type,
                "NodeId": nodeid_str,
                "BrowseName": self.qname_to_string(browse_name),
            }
            if parent_nodeid:
                attrs["ParentNodeId"] = parent_nodeid

            try:
                value_rank = node.get_value_rank()
                if value_rank is not None and value_rank != -1:
                    attrs["ValueRank"] = str(value_rank)
            except Exception:
                pass

            access = self.get_access_level_int(node, ua.AttributeIds.AccessLevel)
            user_access = self.get_access_level_int(node, ua.AttributeIds.UserAccessLevel)
            if access is not None:
                attrs["AccessLevel"] = str(access)
            if user_access is not None:
                attrs["UserAccessLevel"] = str(user_access)

            element = SubElement(root, "UAVariable", attrs)

        elif node_class == ua.NodeClass.Method:
            attrs = {
                "NodeId": nodeid_str,
                "BrowseName": self.qname_to_string(browse_name),
            }
            if parent_nodeid:
                attrs["ParentNodeId"] = parent_nodeid
            element = SubElement(root, "UAMethod", attrs)

        else:
            return

        self.build_display_name_element(element, node)
        self.build_description_element(element, node)
        self.build_references_element(element, node)

        if node_class == ua.NodeClass.Variable:
            try:
                value = node.get_value()
                self.add_typed_value_element(element, element.attrib.get("DataType", ""), value)
            except Exception:
                pass

        try:
            for child in node.get_children():
                # Export application/custom address space and companion-model instances under Objects.
                # Avoid full namespace-0 standard tree explosion.
                # if child.nodeid.NamespaceIndex == 0:
                #     continue
                # self.export_node(child, nodeid_str, root, visited)
                if not self.should_export_node(child):
                    continue
                self.export_node(child, nodeid_str, root, visited)

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Build / write XML
    # -------------------------------------------------------------------------

    def build_address_space_xml(self):
        root = self.create_uanodeset_root()

        self.add_namespace_uris(root)
        self.add_models(root)
        # self.load_uax_direct_value_types(self.default_types_xsd)

        # self.load_uax_complex_type_fields(
        #     self.default_types_xsd
        # )

        self.load_uax_direct_value_types(self.default_types_xsd)

        self.uax_complex_type_fields = {}

        for xsd_path in self.model_xsd_paths:
            self.load_uax_complex_type_fields(xsd_path)

        self.load_complex_type_fallback_fields()

        self.add_aliases(root)

        if self.enable_siome_hash_extensions:
            hash_ns1, hash_si = self.generate_extension_hashes()
            self.add_extensions(root, hash_ns1=hash_ns1, hash_si=hash_si)
        else:
            print("[CLIENT] SIOME hash extensions disabled")

        try:
            objects_node = self.client.get_objects_node()
            visited = set()

            for child in objects_node.get_children():
                # Skip namespace-0 standard objects such as Server unless you explicitly want them.
                # if child.nodeid.NamespaceIndex == 0:
                #     continue
                # self.export_node(child, objects_node.nodeid.to_string(), root, visited)
                if not self.should_export_node(child):
                    continue
                self.export_node(child, objects_node.nodeid.to_string(), root, visited)

        except Exception as e:
            print(f"[CLIENT] Address space export error: {e}")

        return root

    def indent_xml(self, element, level=0):
        indent = "    "
        i = "\n" + level * indent
        if len(element):
            if not element.text or not element.text.strip():
                element.text = i + indent
            for child in element:
                self.indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        if level and (not element.tail or not element.tail.strip()):
            element.tail = i

    def rewrite_extension_children_self_closing(self, file_path):
        """Make empty si extension tags self-closing for SIOME-like formatting."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_text = f.read()

            xml_text = re.sub(r"<(si:Generator\b[^>]*)></si:Generator>", r"<\1/>", xml_text)
            xml_text = re.sub(r"<(si:GeneratorExtension\b[^>]*)></si:GeneratorExtension>", r"<\1/>", xml_text)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(xml_text)
        except Exception:
            pass

    def write_address_space_xml(self, output_dir):
        """
        Create a timestamped SIOME-compatible NodeSet XML file automatically.
        No XML filename needs to be passed from main().
        """
        root = self.build_address_space_xml()
        self.indent_xml(root)

        os.makedirs(output_dir, exist_ok=True)
        file_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        file_path = os.path.join(output_dir, f"address_space_{file_timestamp}.xml")

        tree = ElementTree(root)
        # tree.write(
        #     file_path,
        #     encoding="utf-8",
        #     xml_declaration=True,
        #     short_empty_elements=False,
        # )
        tree.write(
            file_path,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=False
        )

        self.rewrite_expected_root_header(file_path)
        self.rewrite_extension_children_self_closing(file_path)

        # self.rewrite_extension_children_self_closing(file_path)

        print(f"[CLIENT] Wrote address space XML: {file_path}")
        return file_path

    def dump_address_space_periodically(self, output_dir, interval_seconds=10, count=2):
        for index in range(count):
            self.write_address_space_xml(output_dir)
            if index < count - 1:
                time.sleep(interval_seconds)

    # -------------------------------------------------------------------------
    # Example sensor reader from your existing workflow
    # -------------------------------------------------------------------------

    def read_sensors(self):
        """Read and display ProductionLine/Sensors values if that structure exists."""
        try:
            objects = self.client.get_objects_node()
            production_line = None

            for child in objects.get_children():
                if child.get_display_name().Text == "ProductionLine":
                    production_line = child
                    break

            if production_line is None:
                print("[CLIENT] ProductionLine object not found; skipping sensor read.")
                return

            sensors_folder = None
            for child in production_line.get_children():
                if child.get_display_name().Text == "Sensors":
                    sensors_folder = child
                    break

            if sensors_folder is None:
                print("[CLIENT] Sensors folder not found; skipping sensor read.")
                return

            print("\n[CLIENT] Reading sensor values...")
            print("─" * 50)

            children = sensors_folder.get_children()
            for sensor_node in children:
                name = sensor_node.get_display_name().Text
                if name.endswith("_Unit"):
                    continue

                try:
                    value = sensor_node.get_value()
                except Exception:
                    continue

                unit = ""
                for unit_node in children:
                    if unit_node.get_display_name().Text == f"{name}_Unit":
                        try:
                            unit = unit_node.get_value()
                        except Exception:
                            unit = ""
                        break

                print(f"  {name}: {value} {unit}")

            print("─" * 50)

        except Exception as e:
            print(f"[CLIENT] Read error: {e}")

    def rewrite_expected_root_header(self, file_path):
        """Force SIOME-compatible root namespaces and prefixes."""

        with open(file_path, "r", encoding="utf-8") as f:
            xml_text = f.read()

        timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        expected_header = (
            f'<UANodeSet LastModified="{timestamp}" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd" '
            'xmlns:uax="http://opcfoundation.org/UA/2008/02/Types.xsd" '
            'xmlns:si="http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions" '
            'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
        )

        # Replace complete UANodeSet start tag
        xml_text = re.sub(
            r"<UANodeSet\b[^>]*>",
            expected_header,
            xml_text,
            count=1
        )

        # Convert auto-generated prefixes to expected SIOME prefixes
        xml_text = xml_text.replace("ns0:GeneratorExtension", "si:GeneratorExtension")
        xml_text = xml_text.replace("ns0:Generator", "si:Generator")
        xml_text = xml_text.replace("ns1:", "uax:")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_text)

    def get_binary_encoding_id(self, data_type_name):
        """
        Find Default Binary encoding dynamically from the datatype node.

        Example:
            EUInformation -> HasEncoding -> Default Binary -> i=888
            StepTraceDataType -> HasEncoding -> Default Binary -> ns=x;i=...
        """

        data_type_nodeid = self.alias_map.get(data_type_name)
        if not data_type_nodeid:
            return ""

        try:
            dt_node = self.client.get_node(data_type_nodeid)

            for ref in dt_node.get_references():
                ref_type = self.reverse_alias_map.get(
                    ref.ReferenceTypeId.to_string(),
                    ref.ReferenceTypeId.to_string()
                )

                if ref_type != "HasEncoding":
                    continue

                enc_node = self.client.get_node(ref.NodeId)
                browse_name = enc_node.get_browse_name().Name

                if "Binary" in browse_name:
                    return ref.NodeId.to_string()

        except Exception:
            pass

        return ""
    
    def get_datatype_definition_fields(self, data_type_name):
        """
        Try to read OPC UA 1.04 DataTypeDefinition from the server.

        Returns field names such as:
            StepTraceDataType -> EncodingMask, NumberOfTracePoints, SamplingInterval, StartTimeOffset
        """

        if data_type_name in self.server_datatype_fields:
            return self.server_datatype_fields[data_type_name]

        fields = []

        data_type_nodeid = self.alias_map.get(data_type_name)
        if not data_type_nodeid:
            self.server_datatype_fields[data_type_name] = fields
            return fields

        try:
            dt_node = self.client.get_node(data_type_nodeid)

            dv = dt_node.get_attribute(ua.AttributeIds.DataTypeDefinition)
            definition = dv.Value.Value

            # StructureDefinition usually has Fields
            if hasattr(definition, "Fields"):
                for field in definition.Fields:
                    if hasattr(field, "Name") and field.Name:
                        fields.append(field.Name)

            # Some stacks may expose structure fields differently
            elif hasattr(definition, "fields"):
                for field in definition.fields:
                    if hasattr(field, "Name") and field.Name:
                        fields.append(field.Name)

        except Exception:
            pass

        self.server_datatype_fields[data_type_name] = fields
        return fields
    
    def load_complex_type_fallback_fields(self):
        """
        Fallback when companion Types.xsd files are missing.
        Keeps SIOME-compatible structure field names for known IJT/SIOME datatypes.
        """

        loaded = 0

        for type_name, fields in self.complex_type_fallback_fields.items():
            if type_name not in self.uax_complex_type_fields:
                self.uax_complex_type_fields[type_name] = fields
                loaded += 1

        print(f"[CLIENT] Loaded {loaded} fallback complex SIOME/IJT type definitions")

    def should_export_node(self, node):
        try:
            if node.nodeid.NamespaceIndex == 0:
                return False

            if self.export_namespace_uris is None:
                return True

            ns_array = self.client.get_namespace_array()
            ns_uri = ns_array[node.nodeid.NamespaceIndex]

            return ns_uri in self.export_namespace_uris

        except Exception:
            return False


# -----------------------------------------------------------------------------
# Main workflow exactly in the style you requested
# -----------------------------------------------------------------------------

def main():
    """Main client workflow."""

    endpoint = "opc.tcp://127.0.0.1:4840/siome_sample/server/"
    client = ProductionLineClient(endpoint)
    client.enable_siome_hash_extensions = False

    print("[CLIENT] Production Line OPCUA Client")
    print(f"[CLIENT] Endpoint: {endpoint}")
    print()

    try:
        # Step 1: Connect
        print("[CLIENT] Step 1: CONNECT")
        print("─" * 50)
        if not client.connect():
            print("[CLIENT] Failed to connect. Make sure the server is running.")
            return
        print()

        # Step 1b: Export Address Space
        print("[CLIENT] Step 1b: EXPORT ADDRESS SPACE")
        print("─" * 50)
        client.dump_address_space_periodically(
            output_dir="client/address_space_exports",
            interval_seconds=10,
            count=2,
        )
        print()

        # Step 2: Browse
        print("[CLIENT] Step 2: BROWSE")
        print("─" * 50)
        print("[CLIENT] Namespace structure:")
        client.browse_namespace()
        client.dump_browse_namespace_to_file(output_dir="client/address_space_exports")
        print()

        print("[CLIENT] Step 3: RUN FOREVER")
        print("─" * 50)
        try:
            while True:
                client.read_sensors()
                client.write_address_space_xml(output_dir="client/address_space_exports")
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n[CLIENT] Interrupted by user")
            return

    except KeyboardInterrupt:
        print("\n[CLIENT] Interrupted by user")

    except Exception as e:
        print(f"[CLIENT] Error: {e}")

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
