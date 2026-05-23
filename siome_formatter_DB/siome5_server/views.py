from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from .models import OPCUAServer
from .siome5_server_exporter import export_nodeset_from_server


@require_http_methods(["GET"])
def export_nodes_xml(request, server_id: int):
    server = get_object_or_404(OPCUAServer, pk=server_id)

    xml_data = export_nodeset_from_server(server.endpoint)

    response = HttpResponse(xml_data, content_type="application/xml")

    response["Content-Disposition"] = (
        f'attachment; filename="siome5_{server.name.replace(" ", "_")}.xml"'
    )

    return response