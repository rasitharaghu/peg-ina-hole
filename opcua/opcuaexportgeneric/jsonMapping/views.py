from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.conf import settings
from django.views.decorators.http import require_http_methods
from pathlib import Path

from .models import OPCUAServer
from .siome5_generic_exporter import (
    export_siome5_with_mapping,
    auto_create_mapping_from_server,
)


@require_http_methods(["GET"])
def export_nodes_xml(request, server_id: int):
    server = get_object_or_404(OPCUAServer, pk=server_id)

    template_path = Path(settings.BASE_DIR) / "AtlasCopco-Tools.Nodeset2.xml"

    mapping_path = Path(settings.BASE_DIR) / "mapping_ur10e_to_siome5.json"

    if mapping_path.exists():
        mapping_source = str(mapping_path)
    else:
        mapping_source = auto_create_mapping_from_server(
            endpoint_url=server.endpoint,
            template_path=str(template_path),
            save_path=str(Path(settings.BASE_DIR) / "generated_auto_mapping.json"),
        )

    xml_data = export_siome5_with_mapping(
        endpoint_url=server.endpoint,
        template_path=str(template_path),
        mapping=mapping_source,
    )

    response = HttpResponse(xml_data, content_type="application/xml")
    response["Content-Disposition"] = (
        f'attachment; filename="siome5_{server.name.replace(" ", "_")}.xml"'
    )

    return response