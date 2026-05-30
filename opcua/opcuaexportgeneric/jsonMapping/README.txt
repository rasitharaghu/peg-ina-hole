These UR10e NodeIds are placeholders. Replace them with actual NodeIds from your UR10e OPC UA server.

Run with manual mapping:

python main_export_siome5.py \
  --endpoint opc.tcp://192.168.0.20:4840 \
  --template AtlasCopco-Tools.Nodeset2.xml \
  --mapping mapping_ur10e_to_siome5.json \
  --output ur10e_siome5.xml

Run with auto-mapping:

python main_export_siome5.py \
  --endpoint opc.tcp://192.168.0.20:4840 \
  --template AtlasCopco-Tools.Nodeset2.xml \
  --output ur10e_siome5.xml \
  --save-auto-mapping generated_ur10e_mapping.json

Your existing exporter is template-driven, so this approach keeps the SIOME5 tree fixed and only injects live values from mapped OPC UA nodes.



DATABASE

So your Django export button will do:

Click Export
   ↓
Get server endpoint from DB
   ↓
Use mapping if available
   ↓
Else auto-map from server
   ↓
Read live OPC UA values
   ↓
Inject values into SIOME5 template
   ↓
Download SIOME5 XML

This matches your current Django flow where OPCUAServer gives the endpoint and export_nodes_xml() returns the XML download.