SOURCES = {'FEMFX': {'code': 'https://github.com/GPUOpen-Effects/FEMFX',
           'revision': '95cf656731a2465ae1d400138170af2d6575b5bb',
           'license': 'MIT; Copyright (c) 2019 Advanced Micro Devices, Inc.',
           'use': 'Unmodified native corotational FEM and stress-threshold facet fracture for numerical '
                  'teacher data',
           'scope': 'Windows CPU reference; numerical-model agreement is distinct from experimental physical '
                    'validation'},
 'ContinuousFlowMatching': {'paper': 'https://arxiv.org/abs/2210.02747',
                            'use': 'Conditional Gaussian paths for irreversible tetrahedron-face separation '
                                   'fields; no optimal transport coupling',
                            'implementation': 'Independent implementation of conditional paths and '
                                              'regression loss; coupled face-field generation uses PyG '
                                              'GATv2, PyTorch attention and external torchdiffeq '
                                              'integration'},
 'torchdiffeq': {'code': 'https://github.com/rtqichen/torchdiffeq',
                 'license': 'MIT',
                 'use': 'External Runge-Kutta integration of the learned field velocity'},
 'ADMM': {'paper': 'https://web.stanford.edu/~boyd/papers/admm_distr_stats.html',
          'use': 'General consensus for engine-free material-corner state correction; fixed-topology convex '
                 'inner solves do not guarantee global nonlinear fracture convergence'},
 'WorldParticle': {'paper': 'https://arxiv.org/abs/2605.15305',
                   'code': 'https://github.com/WorldParticle/WorldParticle',
                   'use': 'Architectural reference for learned Lagrangian prediction-correction; the '
                          'temporal fracture model is independently implemented with PyTorch and PyG',
                   'upstream_code_or_weights_copied': False},
 'ADMMProjectiveDynamics': {'paper': 'https://www.cse.iitd.ac.in/~narain/admm-pd/',
                            'use': 'Prior art for local constraint projections and shared state correction; '
                                   'independent implementation with SciPy and Clarabel'},
 'libraries': {'numerics': ['https://numpy.org/', 'https://scipy.org/'],
               'learning': ['https://pytorch.org/', 'https://github.com/pyg-team/pytorch_geometric'],
               'optimization': 'https://clarabel.org/',
               'rendering': 'https://threejs.org/'}}
