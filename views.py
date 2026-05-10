from pathlib import Path
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods

from .models import OPCUAServer
from .nodeset_template_formatter import generate_nodeset_xml_from_nodes


@require_http_methods(["GET"])
def export_nodes_xml(request, server_id: int):
    server = get_object_or_404(OPCUAServer, pk=server_id)

    nodes = server.nodes.all()

    template_path = (
        Path(settings.BASE_DIR)
        / "AtlasCopco-Tools.Nodeset2_ToBeWantedSample.xml"
    )

    xml_data = generate_nodeset_xml_from_nodes(
        nodes=nodes,
        template_path=str(template_path),
    )

    response = HttpResponse(xml_data, content_type="application/xml")
    response["Content-Disposition"] = (
        f'attachment; filename="nodeset_{server.name.replace(" ", "_")}.xml"'
    )

    return response