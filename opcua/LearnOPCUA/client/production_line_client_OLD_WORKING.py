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

    def add_extensions(self, root,  hash_ns1, hash_si, product0="SiOME", edition0="Sinumerik", version0="2.8.5-installer",
                       product1="SiOME", edition1="Sinumerik", version1="2.8.5-installer"):
        """Add Extensions with Generator elements for si and ns1 namespaces."""
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
        
        # # Third Extension: ns1:GeneratorExtension
        # ext3 = SubElement(extensions_el, "Extension")
        # SubElement(ext3, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
        #     "Hash": hash_ns1
        # })
        
        # # Fourth Extension: si:GeneratorExtension
        # ext4 = SubElement(extensions_el, "Extension")
        # SubElement(ext4, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
        #     "Hash": hash_si
        # })

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
                data_type_nodeid = node.get_data_type().to_string()
                data_type = self.reverse_alias_map.get(data_type_nodeid, data_type_nodeid)
            except Exception:
                data_type = ""

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
                value = node.get_value()
                self.add_typed_value_element(element, data_type, value)
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


    def add_typed_value_element(self, variable_el, data_type_name, value):
        if value is None:
            return

        value_el = SubElement(variable_el, "Value")

        if data_type_name in self.uax_direct_value_types:
            child = SubElement(
                value_el,
                f"{{{self.UAX_NS}}}{data_type_name}"
            )
            child.text = self.format_value_for_xml(value)
            return

        # variable_el.remove(value_el)
        if isinstance(value, (list, tuple)):
            variable_el.remove(value_el)
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

    def generate_extension_hashes(self):
        """
        Generate deterministic hashes from the live server address space.
        """
        structural_buffer = []
        contextual_buffer = []

        try:
            objects_node = self.client.get_objects_node()
            nodes_to_visit = [objects_node]

            while nodes_to_visit:
                node = nodes_to_visit.pop(0)

                try:
                    nodeid = node.nodeid.to_string()
                    browse_name = node.get_browse_name().to_string()
                    display_name = node.get_display_name().Text

                    structural_buffer.append(f"{nodeid}|{browse_name}")
                    contextual_buffer.append(f"{nodeid}|{browse_name}|{display_name}")

                    try:
                        value = node.get_value()
                        contextual_buffer.append(str(value))
                    except Exception:
                        pass

                    nodes_to_visit.extend(node.get_children())

                except Exception:
                    continue

        except Exception as e:
            print(f"[CLIENT] Hash generation warning: {e}")

        ns1_str = "".join(structural_buffer)
        si_str = "".join(contextual_buffer)

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
    
def main():
    """Main client workflow."""
    
    endpoint = "opc.tcp://127.0.0.1:4840"
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
