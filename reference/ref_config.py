
from typing import List


class RefSetting:
    input_attentions = []
    output_attentions = []
    q_mode = False
    k_mode = False
    v_mode = False
    normal_mode = False

    def __init__(self,
                 input_attentions,
                 output_attentions,
                 q_mode,
                 k_mode,
                 v_mode,
                 normal_mode) -> None:
        self.input_attentions = input_attentions
        self.output_attentions = output_attentions
        self.q_mode = q_mode
        self.k_mode = k_mode
        self.v_mode = v_mode
        self.normal_mode = normal_mode


class RefConfig:
    settings = []

    def __init__(self,
                 ref_latents,
                 positive,
                 negative,
                 sampling,
                 start_percent,
                 end_percent,
                 settings: List[RefSetting]):
        self.ref_latents = ref_latents
        self.positive = positive
        self.negative = negative
        self.sampling = sampling
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.settings = settings
