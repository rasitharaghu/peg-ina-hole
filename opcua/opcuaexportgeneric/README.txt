Run it like this:

python main_export_siome5.py \
  --endpoint opc.tcp://localhost:4840 \
  --template AtlasCopco-Tools.Nodeset2.xml \
  --mapping mapping_ur10e_to_siome5.json \
  --output output_siome5.xml