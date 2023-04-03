
PRIMITIVES = {
    # macro level
    "Linear1": [(1,2)],
    "Linear3": [(1,2),(2,3),(3,4)],
    "Linear4": [(1,2),(2,3),(3,4),(4,5),
    "Residual3": [(1, 2), (2, 3), (1, 4), (3, 4)],
    "Diamond3": [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)],
    "Linear2": [(1,2),(2,3)],
    "Residual2": [(1,2),(2,3),(1,3)],
    "Diamond2": [(1,2),(1,3),(3,4),(2,4)],
    "Cell" : [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)]
    # cell level
    "id": ops.Identity(),
    "zero": ops.Zero(stride=1),
    "conv3x3": {
        "op": nb201_ops.ReLUConvBN,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "affine": True,
    },
    "conv1x1": {
        "op": nb201_ops.ReLUConvBN,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "affine": True,
    },
    "avg_pool": {"op": nb201_ops.POOLING, "mode": "avg", "stride": 1, "affine": True},
    "Cell": NASBench201Cell,
    # conv block level
    "conv3x3o": {
        "op": nb201_ops.Conv,
        "kernel_size": 3,
    },
    "conv1x1o": {"op": nb201_ops.Conv, "kernel_size": 1},
    "dconv3x3o": {
        "op": nb201_ops.DepthwiseConv,
        "kernel_size": 3,
    },
    "batch": {"op": nb201_ops.Normalization, "norm_type": "batch_norm"},
    "instance": {
        "op": nb201_ops.Normalization,
        "norm_type": "instance_norm",
    },
    "layer": {"op": nb201_ops.Normalization, "norm_type": "layer_norm"},
    "relu": {"op": nb201_ops.Activation, "act_type": "relu"},
    "hardswish": {"op": nb201_ops.Activation, "act_type": "hardswish"},
    "mish": {"op": nb201_ops.Activation, "act_type": "mish"},
}