"""
Production Line OPCUA Client
Connects to the production line server, browses the namespace, reads sensor values,
and subscribes to real-time updates.

Usage:
    python production_line_client.py
"""

import os
import re
import time
import sys
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement, ElementTree, register_namespace

from opcua import Client, ua
from datetime import datetime, timezone
import hashlib
import json
import xml.etree.ElementTree as ET

class ProductionLineClient:
    """OPCUA Client for the production line."""
    
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.client = Client(endpoint)
        self.alias_map = {}
        self.reverse_alias_map = {}
        self.uax_direct_value_types = set()
    
    def connect(self):
        """Connect to the OPCUA server."""
        try:
            self.client.connect()
            print("[CLIENT] Connected successfully!")
            return True
        except Exception as e:
            print(f"[CLIENT] Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the OPCUA server."""
        try:
            self.client.disconnect()
            print("[CLIENT] Disconnected successfully")
        except Exception as e:
            print(f"[CLIENT] Disconnect error: {e}")
    
    def browse_namespace(self, node=None, indent=0, output_file=None):
        """Recursively browse and print the namespace tree."""
        if node is None:
            node = self.client.get_root_node()
        
        try:
            # Prepare output text
            output_text = "  " * indent + f"├─ {node.get_display_name().Text} (ID: {node.nodeid})"
            
            # Write to file or stdout
            if output_file:
                output_file.write(output_text + "\n")
            else:
                print(output_text)
            
            # Recursively browse children
            children = node.get_children()
            for i, child in enumerate(children):
                is_last = (i == len(children) - 1)
                self.browse_namespace(child, indent + 1, output_file)
                # time.sleep(0.1)  # Slow down browsing for better readability
        
        except Exception as e:
            error_text = f"[CLIENT] Browse error: {e}"
            if output_file:
                output_file.write(error_text + "\n")
            else:
                print(error_text)
    
    def dump_browse_namespace_to_file(self, output_dir):
        """Dump the browse namespace output to a timestamped file."""
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
    
    # def create_uanodeset_root(self):
    #     """Create the root UANodeSet element with namespace declarations."""
    #     register_namespace("", "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd")
    #     register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    #     register_namespace("uax", "http://opcfoundation.org/UA/2008/02/Types.xsd")
    #     register_namespace("si", "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions")
    #     register_namespace("xsd", "http://www.w3.org/2001/XMLSchema")

    #     root = Element("UANodeSet", {
    #         "LastModified": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    #         "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
    #         "xmlns": "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd",
    #         "xmlns:uax": "http://opcfoundation.org/UA/2008/02/Types.xsd",
    #         "xmlns:si": "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions",
    #         "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
    #         "xmlns:ns0": "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd",
    #         "xmlns:ns1": "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions",
    #         "xmlns:ns2": "http://opcfoundation.org/UA/2008/02/Types.xsd",
    #         "xmlns:ns3": "http://ab.com/UA/DI/AMB/Machinery/MachineryResult/IJTBase/AIJT/Types.xsd",
    #         "xmlns:ns4": "http://opcfoundation.org/UA/Machinery/Result/Types.xsd",
    #         "xmlns:ns5": "http://opcfoundation.org/UA/IJT/Base/Types.xsd",
    #     })
    #     return root

    def create_uanodeset_root(self):
        register_namespace("si", "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions")
        register_namespace("uax", "http://opcfoundation.org/UA/2008/02/Types.xsd")

        return Element("UANodeSet", {
            "LastModified": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        })
        
    def rewrite_expected_root_header(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            xml_text = f.read()

        timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        expected_header = (
            f'<UANodeSet LastModified="{timestamp}" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd" '
            'xmlns:uax="http://opcfoundation.org/UA/2008/02/Types.xsd" '
            'xmlns:si="http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions" '
            'xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
            'xmlns:ns0="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd" '
            'xmlns:ns1="http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions" '
            'xmlns:ns2="http://opcfoundation.org/UA/2008/02/Types.xsd" '
            'xmlns:ns3="http://ab.com/UA/DI/AMB/Machinery/MachineryResult/IJTBase/AIJT/Types.xsd" '
            'xmlns:ns4="http://opcfoundation.org/UA/Machinery/Result/Types.xsd" '
            'xmlns:ns5="http://opcfoundation.org/UA/IJT/Base/Types.xsd">'
        )

        xml_text = re.sub(r"<UANodeSet\b[^>]*>", expected_header, xml_text, count=1)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_text)

    def find_child_by_browse_name(self, parent, browse_name):
        """Find a child node by its BrowseName name."""
        try:
            for child in parent.get_children():
                child_browse_name = child.get_browse_name()
                if hasattr(child_browse_name, "Name") and child_browse_name.Name == browse_name:
                    return child
        except Exception:
            pass
        return None

    # def get_child_value(self, parent, child_name):
    #     """Read a child variable value by browse name."""
    #     child = self.find_child_by_browse_name(parent, child_name)
    #     if child is None:
    #         return ""
    #     try:
    #         return str(child.get_value())
    #     except Exception:
    #         return ""


    def get_child_value(self, parent, child_name):
        """Read a child variable value by browse name."""
        child = self.find_child_by_browse_name(parent, child_name)
        if child is None:
            return ""

        try:
            value = child.get_value()

            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)

                return value.isoformat().replace("+00:00", "Z")

            return str(value)

        except Exception:
            return ""

    def add_namespace_uris(self, root):
        """Add NamespaceUris section from the server."""
        namespace_uris = []
        try:
            namespace_uris = self.client.get_namespace_array()
        except Exception:
            try:
                namespace_uris = self.client.uaclient.get_namespace_array()
            except Exception:
                namespace_uris = []

        namespace_uris_el = SubElement(root, "NamespaceUris")
        for uri in namespace_uris:
            SubElement(namespace_uris_el, "Uri").text = str(uri)

    # def add_models(self, root):
    #     """Add Models section from the server Namespaces list."""
    #     models_el = SubElement(root, "Models")
    #     try:
    #         objects_node = self.client.get_objects_node()
    #         server_node = self.find_child_by_browse_name(objects_node, "Server")
    #         if server_node is None:
    #             return

    #         namespaces_node = self.find_child_by_browse_name(server_node, "Namespaces")
    #         if namespaces_node is None:
    #             return

    #         for namespace_item in namespaces_node.get_children():
    #             namespace_uri = self.get_child_value(namespace_item, "NamespaceUri")
    #             namespace_pub_date = self.get_child_value(namespace_item, "NamespacePublicationDate")
    #             namespace_version = self.get_child_value(namespace_item, "NamespaceVersion")

    #             if not namespace_uri:
    #                 continue

    #             SubElement(models_el, "Model", {
    #                 "ModelUri": namespace_uri,
    #                 "PublicationDate": namespace_pub_date,
    #                 "Version": namespace_version,
    #             })
    #     except Exception:
    #         pass

    def add_models(self, root):
        """Add Models section from the server Namespaces list with strict W3C Date sanitization."""
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

                if namespace_pub_date:
                    namespace_pub_date = namespace_pub_date.strip()
                    if " " in namespace_pub_date and "T" not in namespace_pub_date:
                        namespace_pub_date = namespace_pub_date.replace(" ", "T")
                    
                    if not namespace_pub_date.endswith("Z") and "+" not in namespace_pub_date and "-" not in namespace_pub_date[-6:]:
                        namespace_pub_date += "Z"
                else:
                    namespace_pub_date = "2020-04-14T00:00:00Z"

                SubElement(models_el, "Model", {
                    "ModelUri": namespace_uri,
                    "PublicationDate": namespace_pub_date,
                    "Version": namespace_version if namespace_version else "1.04.6",
                })
        except Exception as e:
            print(f"[CLIENT] Model generation warning: {e}")

    # def add_aliases(self, root):
    #     """Add Aliases section from the server if available."""
    #     aliases_el = SubElement(root, "Aliases")
    #     try:
    #         alias_dict = {}
    #         if hasattr(self.client, "uaclient") and hasattr(self.client.uaclient, "alias"):
    #             alias_dict = self.client.uaclient.alias
    #         elif hasattr(self.client, "uaclient") and hasattr(self.client.uaclient, "aliases"):
    #             alias_dict = self.client.uaclient.aliases
    #         elif hasattr(self.client, "get_aliases"):
    #             alias_dict = self.client.get_aliases()

    #         if isinstance(alias_dict, dict):
    #             for alias, nodeid in alias_dict.items():
    #                 alias_el = SubElement(aliases_el, "Alias", {"Alias": str(alias)})
    #                 alias_el.text = str(nodeid)
    #     except Exception:
    #         pass

    def add_aliases(self, root):
        """Dynamically extracts all available types and references from the live server."""
        aliases_el = SubElement(root, "Aliases")
        alias_map = {}

        try:
            type_roots = [24, 58, 62, 31]
            nodes_to_browse = []

            for root_id in type_roots:
                try:
                    nodes_to_browse.append(self.client.get_node(ua.NodeId(root_id, 0)))
                except Exception:
                    continue

            while nodes_to_browse:
                current_node = nodes_to_browse.pop(0)
                try:
                    browse_name = current_node.get_browse_name().Name
                    nodeid_str = current_node.nodeid.to_string()

                    if browse_name and browse_name not in alias_map:
                        if not browse_name.replace('_', '').isdigit():
                            alias_map[browse_name] = nodeid_str

                    children = current_node.get_children()
                    if children:
                        nodes_to_browse.extend(children)
                except Exception:
                    continue

        except Exception as e:
            print(f"[CLIENT] Dynamic alias discovery error: {e}")

        try:
            lib_aliases = {}
            if hasattr(self.client, "uaclient") and hasattr(self.client.uaclient, "alias"):
                lib_aliases = self.client.uaclient.alias
            elif hasattr(self.client, "get_aliases"):
                lib_aliases = self.client.get_aliases()
                
            if isinstance(lib_aliases, dict):
                for k, v in lib_aliases.items():
                    if k not in alias_map:
                        alias_map[str(k)] = str(v)
        except Exception:
            pass

        sorted_aliases = sorted(alias_map.items())

        for alias_name, node_id in sorted_aliases:
            alias_el = SubElement(aliases_el, "Alias", {"Alias": alias_name})
            alias_el.text = node_id

        self.alias_map = alias_map
        self.reverse_alias_map = {nodeid: alias for alias, nodeid in alias_map.items()}

    # def add_extensions(self, root,  hash_ns1, hash_si, product0="SiOME", edition0="Sinumerik", version0="2.8.5-installer",
    #                    product1="SiOME", edition1="Sinumerik", version1="2.8.5-installer"):
    #     """Add Extensions with Generator elements for si and ns1 namespaces."""
    #     extensions_el = SubElement(root, "Extensions")
        
    #     # First Extension: si:Generator
    #     ext1 = SubElement(extensions_el, "Extension")
    #     SubElement(ext1, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}Generator", {
    #         "Product": product0,
    #         "Edition": edition0,
    #         "Version": version0
    #     })
        
    #     # Second Extension: ns1:Generator
    #     ext2 = SubElement(extensions_el, "Extension")
    #     SubElement(ext2, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}Generator", {
    #         "Product": product1,
    #         "Edition": edition1,
    #         "Version": version1
    #     })
        
    #     # Third Extension: ns1:GeneratorExtension
    #     ext3 = SubElement(extensions_el, "Extension")
    #     SubElement(ext3, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
    #         "Hash": hash_ns1
    #     })
        
    #     # Fourth Extension: si:GeneratorExtension
    #     ext4 = SubElement(extensions_el, "Extension")
    #     SubElement(ext4, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
    #         "Hash": hash_si
    #     })
    
    def add_extensions(self, root, hash_ns1, hash_si, product0="SiOME", edition0="Sinumerik", version0="2.8.5-installer",
                       product1="SiOME", edition1="Sinumerik", version1="2.8.5-installer"):
        """Add Extensions with correctly uncommented Generator & GeneratorExtension elements."""
        extensions_el = SubElement(root, "Extensions")
        
        # First Extension: si:Generator
        ext1 = SubElement(extensions_el, "Extension")
        SubElement(ext1, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}Generator", {
            "Product": product0,
            "Edition": edition0,
            "Version": version0
        })
        
        # Second Extension: ns1:Generator
        ext2 = SubElement(extensions_el, "Extension")
        SubElement(ext2, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}Generator", {
            "Product": product1,
            "Edition": edition1,
            "Version": version1
        })
        
        # Third Extension: ns1:GeneratorExtension (Uncommented and active)
        ext3 = SubElement(extensions_el, "Extension")
        SubElement(ext3, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
            "Hash": hash_ns1
        })
        
        # Fourth Extension: si:GeneratorExtension (Uncommented and active)
        ext4 = SubElement(extensions_el, "Extension")
        SubElement(ext4, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
            "Hash": hash_si
        })

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

    # def build_references_element(self, element, node):
    #     references_el = SubElement(element, "References")
    #     try:
    #         references = node.get_references()
    #         for ref in references:
    #             ref_attribs = {
    #                 "ReferenceType": ref.ReferenceTypeId.to_string() if hasattr(ref.ReferenceTypeId, "to_string") else str(ref.ReferenceTypeId),
    #                 "IsForward": str(ref.IsForward).lower(),
    #             }
    #             reference_el = SubElement(references_el, "Reference", ref_attribs)
    #             reference_el.text = str(ref.NodeId)
    #     except Exception:
    #         pass

    def build_references_element(self, element, node):
        references_el = SubElement(element, "References")

        try:
            references = node.get_references()

            # for ref in references:
            #     ref_type_id = ref.ReferenceTypeId.to_string()
            #     ref_type = self.reverse_alias_map.get(ref_type_id, ref_type_id)

            #     ref_attribs = {
            #         "ReferenceType": ref_type,
            #     }

            #     if ref.IsForward is False:
            #         ref_attribs["IsForward"] = "false"

            #     reference_el = SubElement(references_el, "Reference", ref_attribs)

            #     if hasattr(ref.NodeId, "to_string"):
            #         reference_el.text = ref.NodeId.to_string()
            #     else:
            #         reference_el.text = str(ref.NodeId)

            allowed_forward_refs = {
                "HasTypeDefinition",
                "HasInterface",
                "HasAddIn",
            }

            for ref in references:
                ref_type_id = ref.ReferenceTypeId.to_string()
                ref_type = self.reverse_alias_map.get(ref_type_id, ref_type_id)

                # Skip forward child references
                if ref.IsForward and ref_type not in allowed_forward_refs:
                    continue

                ref_attribs = {
                    "ReferenceType": ref_type,
                }

                if ref.IsForward is False:
                    ref_attribs["IsForward"] = "false"

                reference_el = SubElement(references_el, "Reference", ref_attribs)

                if hasattr(ref.NodeId, "to_string"):
                    reference_el.text = ref.NodeId.to_string()
                else:
                    reference_el.text = str(ref.NodeId)

        except Exception:
            pass

    def export_node(self, node, parent_nodeid, root):
        """Export a single node and recurse through children."""
        try:
            node_class = node.get_node_class()
        except Exception:
            return

        if node_class == ua.NodeClass.Object:
            element = SubElement(root, "UAObject", {
                "SymbolicName": self.make_symbolic_name(node.get_browse_name().Name) if hasattr(node.get_browse_name(), "Name") else str(node.get_browse_name()),
                "NodeId": node.nodeid.to_string(),
                "BrowseName": node.get_browse_name().to_string() if hasattr(node.get_browse_name(), "to_string") else str(node.get_browse_name()),
                "ParentNodeId": str(parent_nodeid) if parent_nodeid is not None else "",
            })
            self.build_display_name_element(element, node)
            self.build_description_element(element, node)
            self.build_references_element(element, node)
            self.ensure_type_definition(element, node, node_class)

        # elif node_class == ua.NodeClass.Variable:
        #     data_type = ""
        #     try:
        #         # data_type = node.get_data_type().to_string()
        #         data_type_nodeid = node.get_data_type().to_string()
        #         data_type = self.reverse_alias_map.get(data_type_nodeid, data_type_nodeid)
        #     except Exception:
        #         data_type = ""

        #     element = SubElement(root, "UAVariable", {
        #         "DataType": data_type,
        #         "NodeId": node.nodeid.to_string(),
        #         "BrowseName": node.get_browse_name().to_string() if hasattr(node.get_browse_name(), "to_string") else str(node.get_browse_name()),
        #         "ParentNodeId": str(parent_nodeid) if parent_nodeid is not None else "",
        #     })
        #     self.build_display_name_element(element, node)
        #     self.build_references_element(element, node)

        #     try:
        #         value = node.get_value()
        #         if value is not None:
        #             value_el = SubElement(element, "Value")
        #             value_el.text = str(value)
        #     except Exception:
        #         pass
        elif node_class == ua.NodeClass.Variable:
            data_type = ""
            data_type_nodeid = ""

            try:
                # Try to resolve the DataType NodeId to a friendly type name
                dt_nodeid = node.get_data_type()
                try:
                    dt_node = self.client.get_node(dt_nodeid)
                    bn = dt_node.get_browse_name()
                    if hasattr(bn, "Name"):
                        data_type = bn.Name
                    else:
                        data_type = str(bn)
                except Exception:
                    # Fallback: use alias map or NodeId string
                    dt_sid = dt_nodeid.to_string() if hasattr(dt_nodeid, "to_string") else str(dt_nodeid)
                    data_type = self.reverse_alias_map.get(dt_sid, dt_sid)
            except Exception:
                data_type = ""

            value_for_export = None
            try:
                value_for_export = node.get_value()
                structured_for_export = self.parse_structured_value(value_for_export)
                if structured_for_export and structured_for_export.get("__DataType"):
                    data_type = structured_for_export["__DataType"]
            except Exception:
                value_for_export = None

            attribs = {
                "DataType": data_type,
                "NodeId": node.nodeid.to_string(),
                "BrowseName": node.get_browse_name().to_string()
                    if hasattr(node.get_browse_name(), "to_string")
                    else str(node.get_browse_name()),
                "ParentNodeId": str(parent_nodeid) if parent_nodeid is not None else "",
            }

            try:
                value_rank = node.get_value_rank()
                if value_rank is not None and value_rank != -1:
                    attribs["ValueRank"] = str(value_rank)
            except Exception:
                pass

            element = SubElement(root, "UAVariable", attribs)

            # Order as per UANodeSet style:
            # DisplayName -> Description -> References -> Value
            self.build_display_name_element(element, node)

            # try:
            #     desc = node.get_description()
            #     if desc and desc.Text:
            #         desc_el = SubElement(element, "Description")
            #         desc_el.text = desc.Text
            # except Exception:
            #     pass

            self.build_description_element(element, node)

            self.build_references_element(element, node)

            try:
                if value_for_export is None:
                    value_for_export = node.get_value()
                self.add_typed_value_element(element, data_type, value_for_export)
            except Exception:
                pass
        else:
            # Only export UAObject and UAVariable
            return

        try:
            # children = node.get_children()
            # for child in children:
            #     self.export_node(child, node.nodeid.to_string(), root)

            children = node.get_children()
            for child in children:
                if child.nodeid.NamespaceIndex == 0:
                    continue
                self.export_node(child, node.nodeid.to_string(), root)
        except Exception:
            pass

    def build_address_space_xml(self):
        root = self.create_uanodeset_root()
        self.add_namespace_uris(root)
        self.add_models(root)

        types_xsd_path = os.path.join(os.path.dirname(__file__), "Opc.Ua.Types.xsd")
        if os.path.exists(types_xsd_path):
            self.load_uax_direct_value_types(types_xsd_path)
            self.load_uax_complex_type_fields(types_xsd_path)

        self.add_aliases(root)

        # Generate hashes before adding Extensions
        hash_ns1, hash_si = self.generate_extension_hashes()

        # Extensions must come after Aliases
        self.add_extensions(root, hash_ns1=hash_ns1, hash_si=hash_si)

        # Nodes come after Extensions
        try:
            # objects_node = self.client.get_objects_node()
            # self.export_node(objects_node, None, root)
            objects_node = self.client.get_objects_node()

            for child in objects_node.get_children():
                if child.nodeid.NamespaceIndex == 0:
                    continue

                self.export_node(child, objects_node.nodeid.to_string(), root)
        except Exception as e:
            print(f"[CLIENT] Address space export error: {e}")

        return root

    def indent_xml(self, element, level=0):
        """Indent XML elements for pretty printing."""
        indent = "    "
        i = "\n" + level * indent
        if len(element):
            if not element.text or not element.text.strip():
                element.text = i + indent
            for child in element:
                self.indent_xml(child, level + 1)
            if not element.tail or not element.tail.strip():
                element.tail = i
        else:
            if level and (not element.tail or not element.tail.strip()):
                element.tail = i

    def write_address_space_xml(self, output_dir):
        """Write the current address space export to a timestamped XML file."""
        root = self.build_address_space_xml()
        self.indent_xml(root)
        os.makedirs(output_dir, exist_ok=True)

        file_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
        file_path = os.path.join(output_dir, f"address_space_{file_timestamp}.xml")

        tree = ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
        # self.rewrite_extension_children_self_closing(file_path)
        self.rewrite_expected_root_header(file_path)
        self.rewrite_extension_children_self_closing(file_path)
        print(f"[CLIENT] Wrote address space XML: {file_path}")

        # Validate the written XML against the UANodeSet XSD
        try:
            from .xml_validator import validate_xml_with_xsd, log_validation_error
        except Exception:
            # relative import fallback for script execution context
            from xml_validator import validate_xml_with_xsd, log_validation_error

        xsd_path = os.path.join(os.path.dirname(__file__), "UANodeSet.xsd")
        is_valid, error_text = validate_xml_with_xsd(file_path, xsd_path)
        if not is_valid:
            log_path = os.path.join(os.path.dirname(__file__), "validation_errors.log")
            log_validation_error(log_path, file_path, error_text)
            raise RuntimeError(f"XML validation failed for {file_path}; see {log_path}")

        return file_path
    # def rewrite_extension_children_self_closing(self, file_path):
    #     """Rewrite empty extension child elements to self-closing syntax only for extensions."""
    #     try:
    #         with open(file_path, "r", encoding="utf-8") as f:
    #             xml_text = f.read()

    #         xml_text = re.sub(r"<(si:Generator\b[^>]*)></si:Generator>", r"<\1 />", xml_text)
    #         xml_text = re.sub(r"<(ns1:Generator\b[^>]*)></ns1:Generator>", r"<\1 />", xml_text)
    #         xml_text = re.sub(r"<(si:GeneratorExtension\b[^>]*)></si:GeneratorExtension>", r"<\1 />", xml_text)
    #         xml_text = re.sub(r"<(ns1:GeneratorExtension\b[^>]*)></ns1:GeneratorExtension>", r"<\1 />", xml_text)

    #         with open(file_path, "w", encoding="utf-8") as f:
    #             f.write(xml_text)
    #     except Exception:
    #         pass

    def rewrite_extension_children_self_closing(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            xml_text = f.read()

        # Make all Siemens extension tags self-closing
        xml_text = re.sub(r"<(si:Generator\b[^>]*)></si:Generator>", r"<\1/>", xml_text)
        xml_text = re.sub(r"<(si:GeneratorExtension\b[^>]*)></si:GeneratorExtension>", r"<\1/>", xml_text)

        # Convert the 2nd Generator to ns1:Generator
        xml_text = re.sub(
            r'(<Extension>\s*)<si:Generator Product="SiOME" Edition="Sinumerik" Version="2.8.5-installer"/>(\s*</Extension>\s*<Extension>\s*)<si:Generator',
            r'\1<si:Generator Product="SiOME" Edition="Sinumerik" Version="2.8.5-installer"/>\2<ns1:Generator',
            xml_text,
            count=1
        )

        # Convert the first GeneratorExtension to ns1:GeneratorExtension
        xml_text = re.sub(
            r'<si:GeneratorExtension Hash="([^"]+)"/>',
            r'<ns1:GeneratorExtension Hash="\1"/>',
            xml_text,
            count=1
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_text)

    def dump_address_space_periodically(self, output_dir, interval_seconds=10, count=2):
        """Dump the address space export periodically into XML files."""
        for index in range(count):
            # Let exceptions propagate so validation failures stop the client
            self.write_address_space_xml(output_dir)
            if index < count - 1:
                time.sleep(interval_seconds)

    def read_sensors(self):
        """Read and display sensor values."""
        try:
            # Get the root node and navigate to sensors
            root = self.client.get_root_node()
            objects = self.client.get_objects_node()
            
            print("\n[CLIENT] Reading sensor values...")
            print("─" * 50)
            
            # Find ProductionLine folder
            for child in objects.get_children():
                if child.get_display_name().Text == "ProductionLine":
                    production_line = child
                    
                    # Find Sensors folder
                    for sensors_child in production_line.get_children():
                        if sensors_child.get_display_name().Text == "Sensors":
                            sensors_folder = sensors_child
                            
                            # Read each sensor
                            for sensor_node in sensors_folder.get_children():
                                name = sensor_node.get_display_name().Text
                                value = sensor_node.get_value()
                                
                                # Skip unit variables, only show sensor values
                                if not name.endswith("_Unit"):
                                    # Try to find corresponding unit variable
                                    unit = "unknown"
                                    for unit_node in sensors_folder.get_children():
                                        if unit_node.get_display_name().Text == f"{name}_Unit":
                                            unit = unit_node.get_value()
                                            break
                                    
                                    print(f"  {name}: {value} {unit}")
            
            print("─" * 50)
        
        except Exception as e:
            print(f"[CLIENT] Read error: {e}")

    UAX_NS = "http://opcfoundation.org/UA/2008/02/Types.xsd"

    def format_value_for_xml(self, value):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat().replace("+00:00", "Z")

        if isinstance(value, bool):
            return str(value).lower()

        return str(value)

    NUMERIC_UAX_TYPES = {
        "Int64", "Int32", "UInt32", "UInt64", "Double", "Float",
        "SByte", "Byte", "Int16", "UInt16",
    }

    def format_value_for_uax(self, data_type_name, value):
        """Format a python value to the canonical string expected by UAX typed elements.

        Returns a string or None when the value cannot be represented safely.
        """
        if value is None:
            return None

        # Strings that are empty should be treated as missing for numeric types
        if isinstance(value, str) and value.strip() == "":
            if data_type_name in self.NUMERIC_UAX_TYPES:
                return None

        # DateTime
        if data_type_name == "DateTime":
            try:
                return self.format_value_for_xml(value)
            except Exception:
                return None

        # Boolean
        if data_type_name == "Boolean":
            try:
                return str(bool(value)).lower()
            except Exception:
                return None

        # Floating point types
        if data_type_name in ("Double", "Float"):
            try:
                return str(float(value))
            except Exception:
                return None

        # Integer-like types
        if data_type_name in self.NUMERIC_UAX_TYPES:
            try:
                return str(int(value))
            except Exception:
                return None

        # Fallback to generic formatter
        try:
            return self.format_value_for_xml(value)
        except Exception:
            return None


    # def add_typed_value_element(self, variable_el, data_type_name, value):
    #     if value is None:
    #         return

    #     value_el = SubElement(variable_el, "Value")

    #     if data_type_name in self.uax_direct_value_types:
    #         child = SubElement(
    #             value_el,
    #             f"{{{self.UAX_NS}}}{data_type_name}"
    #         )
    #         child.text = self.format_value_for_xml(value)
    #         return

    #     # variable_el.remove(value_el)
    #     if isinstance(value, (list, tuple)):
    #         variable_el.remove(value_el)
    #         return

    def add_typed_value_element(self, variable_el, data_type_name, value):
        if value is None:
            return

        # Lists should be handled separately, unless the variable value itself is a
        # JSON/dict structure that describes a list field.
        if isinstance(value, (list, tuple)):
            return

        structured = self.parse_structured_value(value)

        # Structured JSON may explicitly define the intended datatype.
        if structured and structured.get("__DataType"):
            data_type_name = structured["__DataType"]
            variable_el.attrib["DataType"] = data_type_name

        # Complex/ExtensionObject types.
        # This is generic: field order comes from Opc.Ua.Types.xsd when known,
        # otherwise from the JSON/dict value itself.
        if structured or data_type_name in self.uax_complex_type_fields:
            if self.add_extension_object_value_element(variable_el, data_type_name, value):
                return

        # Simple scalar types only
        if data_type_name in self.uax_direct_value_types and data_type_name not in self.uax_complex_type_fields:
            formatted = self.format_value_for_uax(data_type_name, value)
            if formatted is None:
                return

            value_el = SubElement(variable_el, "Value")
            child = SubElement(value_el, f"{{{self.UAX_NS}}}{data_type_name}")
            child.text = formatted
            return

        # Unknown/custom datatype: avoid invalid XML
        return

    def load_uax_direct_value_types(self, types_xsd_path):
        """
        Load valid direct OPC UA Types.xsd value elements dynamically.
        Example: String, Double, Boolean, DateTime, LocalizedText, etc.
        """
        import xml.etree.ElementTree as ET

        xsd_ns = "{http://www.w3.org/2001/XMLSchema}"
        tree = ET.parse(types_xsd_path)
        root = tree.getroot()

        value_types = set()

        for element in root.findall(f".//{xsd_ns}element"):
            name = element.attrib.get("name")
            if name:
                value_types.add(name)

        self.uax_direct_value_types = value_types

    # def generate_extension_hashes(self):
    #     """
    #     Generate deterministic hashes from the live server address space.
    #     """
    #     structural_buffer = []
    #     contextual_buffer = []

    #     try:
    #         objects_node = self.client.get_objects_node()
    #         nodes_to_visit = [objects_node]

    #         while nodes_to_visit:
    #             node = nodes_to_visit.pop(0)

    #             try:
    #                 nodeid = node.nodeid.to_string()
    #                 browse_name = node.get_browse_name().to_string()
    #                 display_name = node.get_display_name().Text

    #                 structural_buffer.append(f"{nodeid}|{browse_name}")
    #                 contextual_buffer.append(f"{nodeid}|{browse_name}|{display_name}")

    #                 try:
    #                     value = node.get_value()
    #                     contextual_buffer.append(str(value))
    #                 except Exception:
    #                     pass

    #                 nodes_to_visit.extend(node.get_children())

    #             except Exception:
    #                 continue

    #     except Exception as e:
    #         print(f"[CLIENT] Hash generation warning: {e}")

    #     ns1_str = "".join(structural_buffer)
    #     si_str = "".join(contextual_buffer)

    #     hash_ns1 = hashlib.md5(ns1_str.encode("utf-8")).hexdigest()
    #     hash_si = hashlib.md5(si_str.encode("utf-8")).hexdigest()

    #     return hash_ns1, hash_si
    
    def generate_extension_hashes(self):
        """
        Generates deterministic cryptographic hashes from the live server address space.
        Implements cyclic tracking to prevent infinite tree-crawling loops.
        """
        structural_buffer = []
        contextual_buffer = []
        
        # Tracking set to remember nodes we've already visited (Prevents infinite loops)
        visited_nodes = set()

        try:
            objects_node = self.client.get_objects_node()
            nodes_to_visit = [objects_node]

            while nodes_to_visit:
                node = nodes_to_visit.pop(0)
                nodeid_str = node.nodeid.to_string()
                
                # If we have already crawled this exact node, skip it
                if nodeid_str in visited_nodes:
                    continue
                visited_nodes.add(nodeid_str)

                # Skip core Namespace 0 components to ensure identical hashes across different server versions
                if node.nodeid.NamespaceIndex == 0 and nodeid_str != "i=85":
                    continue

                try:
                    browse_name = node.get_browse_name().to_string() if hasattr(node.get_browse_name(), "to_string") else str(node.get_browse_name())
                    display_name = node.get_display_name().Text if hasattr(node.get_display_name(), "Text") else str(node.get_display_name())

                    # Clean up strings by stripping random whitespace
                    browse_name_clean = browse_name.strip()
                    display_name_clean = display_name.strip()

                    # Append to tracking buffers
                    structural_buffer.append(f"{nodeid_str}|{browse_name_clean}")
                    contextual_buffer.append(f"{nodeid_str}|{browse_name_clean}|{display_name_clean}")

                    # Attempt to pull dynamic value metrics for the SI contextual hash signature
                    try:
                        value = node.get_value()
                        if value is not None:
                            contextual_buffer.append(self.format_value_for_xml(value))
                    except Exception:
                        pass

                    # Safely extend the crawling queue with child objects
                    for child in node.get_children():
                        if child.nodeid.to_string() not in visited_nodes:
                            nodes_to_visit.append(child)

                except Exception:
                    continue

        except Exception as e:
            print(f"[CLIENT] Dynamic hash generation warning: {e}")

        # Combine items into unified byte strings
        ns1_str = "".join(structural_buffer)
        si_str = "".join(contextual_buffer)

        # Convert to standard lowercase MD5 hex digests expected by SiOME
        hash_ns1 = hashlib.md5(ns1_str.encode("utf-8")).hexdigest()
        hash_si = hashlib.md5(si_str.encode("utf-8")).hexdigest()

        return hash_ns1, hash_si
    
    def make_symbolic_name(self, name):
        if not name:
            return ""

        name = re.sub(r"[^A-Za-z0-9_]", "_", str(name))

        if not re.match(r"^[A-Za-z_]", name):
            name = "S" + name

        return name
    
    def load_uax_complex_type_fields(self, types_xsd_path):

        XS = "{http://www.w3.org/2001/XMLSchema}"
        tree = ET.parse(types_xsd_path)
        root = tree.getroot()

        self.uax_complex_type_fields = {}

        for complex_type in root.findall(f".//{XS}complexType"):
            type_name = complex_type.attrib.get("name")
            if not type_name:
                continue

            # Example: EUInformation -> EUInformation
            if type_name.endswith("DataType"):
                element_name = type_name.replace("DataType", "")
            else:
                element_name = type_name

            fields = []

            for field in complex_type.findall(f".//{XS}element"):
                field_name = field.attrib.get("name")
                if field_name:
                    fields.append(field_name)

            if fields:
                self.uax_complex_type_fields[element_name] = fields

    def parse_structured_value(self, value):
        """Return dict if value is a JSON encoded structured value, else None.

        The dummy server can publish complex values as JSON strings, for example:
        {
          "__DataType": "EUInformation",
          "__TypeId": "i=888",
          "NamespaceUri": "...",
          "UnitId": 4279624,
          "DisplayName": {"Text": "A·h"},
          "Description": {"Text": "ampere hour"}
        }

        This keeps the client generic: the datatype name, encoding id and fields
        come from the value/server, while the XML field order still comes from
        Opc.Ua.Types.xsd when available.
        """
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None

        return None

    def add_localized_text_value(self, parent_el, localized_value):
        """Create UAX LocalizedText-like content."""
        if localized_value is None:
            return

        if hasattr(localized_value, "Locale") and localized_value.Locale:
            locale_el = SubElement(parent_el, f"{{{self.UAX_NS}}}Locale")
            locale_el.text = str(localized_value.Locale)

        if hasattr(localized_value, "Text"):
            text = localized_value.Text
        elif isinstance(localized_value, dict):
            text = localized_value.get("Text", "")
        else:
            text = str(localized_value)

        text_el = SubElement(parent_el, f"{{{self.UAX_NS}}}Text")
        text_el.text = "" if text is None else str(text)

    def add_generic_field_value(self, parent_el, field_name, field_value):
        """Serialize a field value under an already-created UAX field element."""
        if field_value is None:
            return False

        # LocalizedText-like dict: {"Text": "...", "Locale": "..."}
        if isinstance(field_value, dict):
            if "Text" in field_value or "Locale" in field_value:
                self.add_localized_text_value(parent_el, field_value)
                return True

            # Nested structure fallback
            for key, val in field_value.items():
                if key.startswith("__"):
                    continue
                child = SubElement(parent_el, f"{{{self.UAX_NS}}}{key}")
                self.add_generic_field_value(child, key, val)
            return True

        # python-opcua LocalizedText
        if hasattr(field_value, "Text"):
            self.add_localized_text_value(parent_el, field_value)
            return True

        # List support
        if isinstance(field_value, (list, tuple)):
            for item in field_value:
                item_el = SubElement(parent_el, f"{{{self.UAX_NS}}}{field_name}")
                self.add_generic_field_value(item_el, field_name, item)
            return True

        formatted = self.format_value_for_xml(field_value)
        if formatted is None:
            return False

        parent_el.text = formatted
        return True

    # def add_extension_object_value_element(self, variable_el, data_type_name, value):
    #     """Add <Value><uax:ExtensionObject>...</...></Value> generically.

    #     Field order is taken from Opc.Ua.Types.xsd when the datatype exists there.
    #     For custom types not in Opc.Ua.Types.xsd, the JSON/dict field order is used.
    #     The binary encoding id comes from:
    #     1) JSON key "__TypeId" / "__BinaryEncodingId"
    #     2) live server HasEncoding -> Default Binary, when available
    #     """
    #     structured = self.parse_structured_value(value)

    #     if structured is None:
    #         # Support real python-opcua structure objects
    #         structured = {}
    #         for field_name in self.uax_complex_type_fields.get(data_type_name, []):
    #             field_value = getattr(value, field_name, None)
    #             if field_value is not None:
    #                 structured[field_name] = field_value

    #     if not structured:
    #         return False

    #     actual_type = structured.get("__DataType", data_type_name)
    #     # type_id_text = (
    #     #     structured.get("__TypeId")
    #     #     or structured.get("__BinaryEncodingId")
    #     #     or self.get_binary_encoding_id(actual_type)
    #     # )

    #     type_id_text = (
    #     structured.get("__TypeId")
    #     or structured.get("__BinaryEncodingId")
    #     )

    #     if not self.is_valid_nodeid_text(type_id_text):
    #         type_id_text = self.get_binary_encoding_id(actual_type)

    #     if not self.is_valid_nodeid_text(type_id_text):
    #         return False

    #     # if not type_id_text:
    #     #     # Do not create invalid <uax:Identifier></uax:Identifier>
    #     #     return False

    #     # Prefer XSD field order, fallback to JSON keys.
    #     field_names = self.uax_complex_type_fields.get(actual_type)
    #     if not field_names:
    #         field_names = [
    #             k for k in structured.keys()
    #             if not k.startswith("__")
    #         ]

    #     value_el = SubElement(variable_el, "Value")
    #     ext_obj = SubElement(value_el, f"{{{self.UAX_NS}}}ExtensionObject")

    #     type_id = SubElement(ext_obj, f"{{{self.UAX_NS}}}TypeId")
    #     SubElement(type_id, f"{{{self.UAX_NS}}}Identifier").text = str(type_id_text)

    #     body = SubElement(ext_obj, f"{{{self.UAX_NS}}}Body")
    #     # complex_el = SubElement(body, f"{{{self.UAX_NS}}}{actual_type}")
    #     body_ns = self.get_datatype_body_namespace(actual_type)
    #     complex_el = SubElement(body, f"{{{body_ns}}}{actual_type}")

    #     written = False
    #     for field_name in field_names:
    #         if field_name.startswith("__"):
    #             continue

    #         if field_name not in structured:
    #             continue

    #         field_value = structured.get(field_name)
    #         if field_value is None:
    #             continue

    #         # field_el = SubElement(complex_el, f"{{{self.UAX_NS}}}{field_name}")
    #         field_el = SubElement(complex_el, f"{{{body_ns}}}{field_name}")
    #         if self.add_generic_field_value(field_el, field_name, field_value):
    #             written = True
    #         else:
    #             complex_el.remove(field_el)

    #     if not written:
    #         variable_el.remove(value_el)
    #         return False

    #     return True

    def add_extension_object_value_element(self, variable_el, data_type_name, value):
        structured = self.parse_structured_value(value)

        if structured is None:
            structured = {}
            for field_name in self.uax_complex_type_fields.get(data_type_name, []):
                field_value = getattr(value, field_name, None)
                if field_value is not None:
                    structured[field_name] = field_value

        if not structured:
            return False

        actual_type = structured.get("__DataType", data_type_name)

        type_id_text = (
            structured.get("__TypeId")
            or structured.get("__BinaryEncodingId")
        )

        if not self.is_valid_nodeid_text(type_id_text):
            type_id_text = self.get_binary_encoding_id(actual_type)

        if not self.is_valid_nodeid_text(type_id_text):
            return False

        field_names = self.uax_complex_type_fields.get(actual_type)
        if not field_names:
            field_names = [
                k for k in structured.keys()
                if not k.startswith("__")
            ]

        value_el = SubElement(variable_el, "Value")
        ext_obj = SubElement(value_el, f"{{{self.UAX_NS}}}ExtensionObject")

        type_id = SubElement(ext_obj, f"{{{self.UAX_NS}}}TypeId")
        SubElement(type_id, f"{{{self.UAX_NS}}}Identifier").text = str(type_id_text)

        body = SubElement(ext_obj, f"{{{self.UAX_NS}}}Body")

        body_ns = self.get_datatype_body_namespace(actual_type)
        complex_el = SubElement(body, f"{{{body_ns}}}{actual_type}")

        written = False

        for field_name in field_names:
            if field_name.startswith("__"):
                continue

            if field_name not in structured:
                continue

            field_value = structured.get(field_name)
            if field_value is None:
                continue

            field_el = SubElement(complex_el, f"{{{body_ns}}}{field_name}")

            if self.add_generic_field_value(field_el, field_name, field_value):
                written = True
            else:
                complex_el.remove(field_el)

        if not written:
            variable_el.remove(value_el)
            return False

        return True

    # def get_binary_encoding_id(self, data_type_name):
    #     """Try to find HasEncoding -> Default Binary for a datatype from the server.

    #     This avoids datatype-specific hardcoding. If the server does not expose the
    #     datatype encoding node, the caller should skip the ExtensionObject value
    #     or supply "__TypeId" in the structured test value.
    #     """
    #     try:
    #         # Resolve datatype node from alias map first.
    #         nodeid_text = self.alias_map.get(data_type_name)
    #         if nodeid_text:
    #             dt_node = self.client.get_node(nodeid_text)
    #         else:
    #             # fallback: browse DataTypes tree and match BrowseName
    #             dt_node = None
    #             queue = [self.client.get_node(ua.NodeId(24, 0))]  # BaseDataType
    #             visited = set()

    #             while queue:
    #                 candidate = queue.pop(0)
    #                 sid = candidate.nodeid.to_string()
    #                 if sid in visited:
    #                     continue
    #                 visited.add(sid)

    #                 try:
    #                     if candidate.get_browse_name().Name == data_type_name:
    #                         dt_node = candidate
    #                         break
    #                     queue.extend(candidate.get_children())
    #                 except Exception:
    #                     continue

    #             if dt_node is None:
    #                 return ""

    #         refs = dt_node.get_references()
    #         for ref in refs:
    #             try:
    #                 ref_type_id = ref.ReferenceTypeId.to_string()
    #                 ref_type = self.reverse_alias_map.get(ref_type_id, ref_type_id)
    #                 if ref_type != "HasEncoding":
    #                     continue
    #                 enc_node = self.client.get_node(ref.NodeId)
    #                 browse_name = enc_node.get_browse_name().Name
    #                 if browse_name == "Default Binary":
    #                     return ref.NodeId.to_string() if hasattr(ref.NodeId, "to_string") else str(ref.NodeId)
    #             except Exception:
    #                 continue

    #     except Exception:
    #         pass

    #     return ""

    def get_binary_encoding_id(self, data_type_name):
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

                if ref_type == "HasEncoding":
                    target = self.client.get_node(ref.NodeId)
                    browse_name = target.get_browse_name().Name

                    if browse_name in ("Default Binary", "DefaultBinary"):
                        return ref.NodeId.to_string()

        except Exception:
            pass

        return ""
    

    def ensure_type_definition(self, element, node, node_class):
        refs_el = element.find("References")
        if refs_el is None:
            refs_el = SubElement(element, "References")

        for ref in refs_el.findall("Reference"):
            if ref.attrib.get("ReferenceType") == "HasTypeDefinition":
                return

        try:
            type_def = node.get_type_definition().to_string()
        except Exception:
            type_def = "i=58" if node_class == ua.NodeClass.Object else "i=63"

        ref_el = SubElement(refs_el, "Reference", {
            "ReferenceType": "HasTypeDefinition"
        })
        ref_el.text = type_def

    def is_valid_nodeid_text(self, text):
        if not text:
            return False
        text = str(text).strip()
        return (
            text.startswith("i=")
            or text.startswith("ns=")
            or text.startswith("g=")
            or text.startswith("s=")
        )
    def get_datatype_body_namespace(self, data_type_name):
        try:
            data_type_nodeid = self.alias_map.get(data_type_name)
            if not data_type_nodeid:
                return self.UAX_NS

            nodeid = ua.NodeId.from_string(data_type_nodeid)
            ns_index = nodeid.NamespaceIndex
            namespace_array = self.client.get_namespace_array()
            model_uri = namespace_array[ns_index]

            if model_uri == "http://opcfoundation.org/UA/":
                return self.UAX_NS

            return model_uri.rstrip("/") + "/Types.xsd"
        except Exception:
            return self.UAX_NS
    
def main():
    """Main client workflow."""
    
    endpoint = "opc.tcp://127.0.0.1:4840/siome_sample/server/"
    client = ProductionLineClient(endpoint)
    
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
