from dataclasses import dataclass


@dataclass
class Context:
    bounces: int
    spp: int
    sample_generation_strategy: str = 'light_tracing'
    iterations: int = 100
    num_samples: int = 0
    rr_alpha: float = 0.8
    width: int = 0
    height: int = 0
    clip_error_calcs: bool = False
    tonemap_error_calc: bool = True, 
    gammacorrect_error_calc: bool = True, 
    track_error: bool = True
    track_error_every: int = 50
    test_spp = 4
    bsdf_sampling: bool = True
    emitter_sampling: bool = True
    mis_beta: float = 2.0
    model_sample_MIS: bool = True
    model_MIS_beta: float = 2.0
    MIS_aware_training: bool = False
    include_brdf: bool = True
    sample_model: bool = False
    only_direct: bool = False
    only_indirect: bool = False
    bsdf_prior: bool = False
    only_optimize_encoding: bool = False