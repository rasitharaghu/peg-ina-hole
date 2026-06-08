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


class ProductionLineClient:
    """OPCUA Client for the production line."""
    
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.client = Client(endpoint)
    
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
    
    def create_uanodeset_root(self):
        """Create the root UANodeSet element with namespace declarations."""
        register_namespace("", "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd")
        register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
        register_namespace("uax", "http://opcfoundation.org/UA/2008/02/Types.xsd")
        register_namespace("si", "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions")
        register_namespace("xsd", "http://www.w3.org/2001/XMLSchema")

        root = Element("UANodeSet", {
            "LastModified": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xmlns": "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd",
            "xmlns:uax": "http://opcfoundation.org/UA/2008/02/Types.xsd",
            "xmlns:si": "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions",
            "xmlns:xsd": "http://www.w3.org/2001/XMLSchema",
            "xmlns:ns0": "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd",
            "xmlns:ns1": "http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions",
            "xmlns:ns2": "http://opcfoundation.org/UA/2008/02/Types.xsd",
            "xmlns:ns3": "http://ab.com/UA/DI/AMB/Machinery/MachineryResult/IJTBase/AIJT/Types.xsd",
            "xmlns:ns4": "http://opcfoundation.org/UA/Machinery/Result/Types.xsd",
            "xmlns:ns5": "http://opcfoundation.org/UA/IJT/Base/Types.xsd",
        })
        return root

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

    def get_child_value(self, parent, child_name):
        """Read a child variable value by browse name."""
        child = self.find_child_by_browse_name(parent, child_name)
        if child is None:
            return ""
        try:
            return str(child.get_value())
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

    def add_models(self, root):
        """Add Models section from the server Namespaces list."""
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

                SubElement(models_el, "Model", {
                    "ModelUri": namespace_uri,
                    "PublicationDate": namespace_pub_date,
                    "Version": namespace_version,
                })
        except Exception:
            pass

    def add_aliases(self, root):
        """Add Aliases section from the server if available."""
        aliases_el = SubElement(root, "Aliases")
        try:
            alias_dict = {}
            if hasattr(self.client, "uaclient") and hasattr(self.client.uaclient, "alias"):
                alias_dict = self.client.uaclient.alias
            elif hasattr(self.client, "uaclient") and hasattr(self.client.uaclient, "aliases"):
                alias_dict = self.client.uaclient.aliases
            elif hasattr(self.client, "get_aliases"):
                alias_dict = self.client.get_aliases()

            if isinstance(alias_dict, dict):
                for alias, nodeid in alias_dict.items():
                    alias_el = SubElement(aliases_el, "Alias", {"Alias": str(alias)})
                    alias_el.text = str(nodeid)
        except Exception:
            pass

    def add_extensions(self, root, product0="SiOME", edition0="Sinumerik", version0="2.8.5-installer",
                       product1="SiOME", edition1="Sinumerik", version1="2.8.5-installer",
                       hash1="b929aa38d5a80048ba9c44c5996fc044", hash2="3649071798aa7e23c0cc2a52e739b463"):
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
        
        # Third Extension: ns1:GeneratorExtension
        ext3 = SubElement(extensions_el, "Extension")
        SubElement(ext3, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
            "Hash": hash1
        })
        
        # Fourth Extension: si:GeneratorExtension
        ext4 = SubElement(extensions_el, "Extension")
        SubElement(ext4, "{http://www.siemens.com/OPCUA/2017/SimaticNodeSetExtensions}GeneratorExtension", {
            "Hash": hash2
        })

    def build_display_name_element(self, element, node):
        display_name = SubElement(element, "DisplayName")
        try:
            display_name.text = node.get_display_name().Text
        except Exception:
            display_name.text = str(node)

    def build_references_element(self, element, node):
        references_el = SubElement(element, "References")
        try:
            references = node.get_references()
            for ref in references:
                ref_attribs = {
                    "ReferenceType": ref.ReferenceTypeId.to_string() if hasattr(ref.ReferenceTypeId, "to_string") else str(ref.ReferenceTypeId),
                    "IsForward": str(ref.IsForward).lower(),
                }
                reference_el = SubElement(references_el, "Reference", ref_attribs)
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
                "SymbolicName": node.get_browse_name().Name if hasattr(node.get_browse_name(), "Name") else str(node.get_browse_name()),
                "NodeId": node.nodeid.to_string(),
                "BrowseName": node.get_browse_name().to_string() if hasattr(node.get_browse_name(), "to_string") else str(node.get_browse_name()),
                "ParentNodeId": str(parent_nodeid) if parent_nodeid is not None else "",
            })
            self.build_display_name_element(element, node)
            self.build_references_element(element, node)

        elif node_class == ua.NodeClass.Variable:
            data_type = ""
            try:
                data_type = node.get_data_type().to_string()
            except Exception:
                data_type = ""

            element = SubElement(root, "UAVariable", {
                "DataType": data_type,
                "NodeId": node.nodeid.to_string(),
                "BrowseName": node.get_browse_name().to_string() if hasattr(node.get_browse_name(), "to_string") else str(node.get_browse_name()),
                "ParentNodeId": str(parent_nodeid) if parent_nodeid is not None else "",
            })
            self.build_display_name_element(element, node)
            self.build_references_element(element, node)

            try:
                value = node.get_value()
                if value is not None:
                    value_el = SubElement(element, "Value")
                    value_el.text = str(value)
            except Exception:
                pass
        else:
            # Only export UAObject and UAVariable
            return

        try:
            children = node.get_children()
            for child in children:
                self.export_node(child, node.nodeid.to_string(), root)
        except Exception:
            pass

    def build_address_space_xml(self):
        """Build the complete UANodeSet XML tree for the current address space."""
        root = self.create_uanodeset_root()
        self.add_namespace_uris(root)
        self.add_models(root)
        self.add_aliases(root)
        self.add_extensions(root)

        try:
            objects_node = self.client.get_objects_node()
            self.export_node(objects_node, None, root)
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
        self.rewrite_extension_children_self_closing(file_path)
        print(f"[CLIENT] Wrote address space XML: {file_path}")

    def rewrite_extension_children_self_closing(self, file_path):
        """Rewrite empty extension child elements to self-closing syntax only for extensions."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_text = f.read()

            xml_text = re.sub(r"<(si:Generator\b[^>]*)></si:Generator>", r"<\1 />", xml_text)
            xml_text = re.sub(r"<(ns1:Generator\b[^>]*)></ns1:Generator>", r"<\1 />", xml_text)
            xml_text = re.sub(r"<(si:GeneratorExtension\b[^>]*)></si:GeneratorExtension>", r"<\1 />", xml_text)
            xml_text = re.sub(r"<(ns1:GeneratorExtension\b[^>]*)></ns1:GeneratorExtension>", r"<\1 />", xml_text)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(xml_text)
        except Exception:
            pass

    def dump_address_space_periodically(self, output_dir, interval_seconds=10, count=2):
        """Dump the address space export periodically into XML files."""
        for index in range(count):
            try:
                self.write_address_space_xml(output_dir)
            except Exception as e:
                print(f"[CLIENT] Periodic dump error: {e}")
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
