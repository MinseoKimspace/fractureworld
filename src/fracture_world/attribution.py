SOURCES = {
    "MeshGraphNets": {
        "paper": "https://arxiv.org/abs/2010.03409",
        "use": "Graph encode-process-decode architectural reference; independently adapted",
        "upstream_code_copied": False,
    },
    "WindLifter": {
        "paper": "https://arxiv.org/abs/2502.00626",
        "code": "https://github.com/Changy1506/WindLifter",
        "use": "External research comparison; no synthetic reproduction or copied source",
    },
    "FElupe": {
        "code": "https://github.com/adtzlr/felupe",
        "use": "Optional external numerical teacher material and finite-element assembly",
    },
    "PyTorchGeometric": {
        "code": "https://github.com/pyg-team/pytorch_geometric",
        "use": "Neural graph message passing and batching",
    },
    "SciPy": {"code": "https://github.com/scipy/scipy", "use": "Topology components and numerical reference integration"},
    "Matplotlib": {"code": "https://github.com/matplotlib/matplotlib", "use": "2D trajectory rendering"},
}
