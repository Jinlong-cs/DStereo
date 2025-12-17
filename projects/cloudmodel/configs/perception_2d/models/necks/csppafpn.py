from hatbc.workflow.trace import GraphTracer

from hat.models.necks.csp_pafpn import CSPPAFPN

channels2blocks = {
    128: 1,
    192: 2,
    256: 3,
    320: 4,
}


def build_neck_csppafpn(
    in_stride2channels, out_stride2channels, node_name="neck"
):
    in_channels = list(in_stride2channels.values())
    out_channel = list(out_stride2channels.values())[0]
    pafpn_neck = dict(
        type="CSPPAFPN",
        in_channels=in_channels,
        out_channels=out_channel,
        num_csp_blocks=channels2blocks[out_channel],
        num_outs=len(out_stride2channels),
        node_name=node_name,
    )
    return dict(
        neck=pafpn_neck,
        stride2channels=out_stride2channels,
    )


# register network in GraphTracer
GraphTracer.register_basic_types(CSPPAFPN)
