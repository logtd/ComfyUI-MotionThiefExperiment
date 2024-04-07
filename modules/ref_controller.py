import enum

from .block_type import BlockType


def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False


def is_temporal_block(module):
    return isinstance_str(module, 'TemporalTransformerBlock')


def is_named_module_transformer_block(named_module):
    if is_temporal_block(named_module[1]):
        return True
    return False


class RefMode(enum.Enum):
    OFF = 'OFF'
    WRITE = 'WRITE'
    READ = 'READ'


class RefController:
    modules = []
    temp_modules = []
    default_video_length = 16
    default_full_length = 16
    model = None

    def add_module(self, module):
        self.modules.append(module)

    def set_mode(self, mode: RefMode):
        for module in self.modules:
            module.ref_mode = mode
        if mode == RefMode.OFF:
            self.set_kv_mode(mode)
            self.set_normal_mode(mode)

    def set_normal_mode(self, mode: RefMode):
        for module in self.modules:
            module.normal_mode = mode

    def set_kv_mode(self, mode: RefMode, count = None):
        for module in self.modules:
            module.kv_mode = RefMode.OFF
        if mode == RefMode.OFF or count is None:
            count = len(self.modules)
        output_modules = list(filter(lambda m: m.block_type == BlockType.OUTPUT, self.modules))
        output_modules = sorted(output_modules, key=lambda m: -m.block_idx)[:count]
        for module in output_modules:
            module.kv_mode = mode

    def set_motion_mode(self, config, mode: RefMode):
        model = self.model
        input_modules = list(filter(is_named_module_transformer_block, model.input_blocks.named_modules()))
        i = 0
        for _, module in input_modules: # 8
            module.attention_blocks[0].mode = 'OFF'
            module.attention_blocks[0].q_bank_enabled = False
            module.attention_blocks[0].k_bank_enabled = False
            module.attention_blocks[0].v_bank_enabled = False
            module.attention_blocks[0].norm_bank_enabled = False
            if len(module.attention_blocks) > 1:
                module.attention_blocks[1].mode = 'OFF'
                module.attention_blocks[1].q_bank_enabled = False
                module.attention_blocks[1].k_bank_enabled = False
                module.attention_blocks[1].v_bank_enabled = False
                module.attention_blocks[1].norm_bank_enabled = False
            if config is not None and config.inputs[i] and mode.name != 'OFF':
                module.attention_blocks[0].mode = mode.name
                module.attention_blocks[0].q_bank_enabled = config.q_bank_enabled
                module.attention_blocks[0].k_bank_enabled = config.k_bank_enabled
                module.attention_blocks[0].v_bank_enabled = config.v_bank_enabled
                module.attention_blocks[0].norm_bank_enabled = config.norm_bank_enabled
                if len(module.attention_blocks) > 1:
                    module.attention_blocks[1].mode = mode.name
                    module.attention_blocks[1].q_bank_enabled = config.q_bank_enabled
                    module.attention_blocks[1].k_bank_enabled = config.k_bank_enabled
                    module.attention_blocks[1].v_bank_enabled = config.v_bank_enabled
                    module.attention_blocks[1].norm_bank_enabled = config.norm_bank_enabled
                if len(module.attention_blocks) > 2:
                    print('whoa in')
            i += 1

        middle_modules = [] # list(filter(is_named_module_transformer_block, model.middle_block.named_modules()))
        output_modules = list(filter(is_named_module_transformer_block, model.output_blocks.named_modules()))

        i = 0
        for _, module in output_modules: # 12
            module.attention_blocks[0].mode = 'OFF'
            module.attention_blocks[0].q_bank_enabled = False
            module.attention_blocks[0].k_bank_enabled = False
            module.attention_blocks[0].v_bank_enabled = False
            module.attention_blocks[0].norm_bank_enabled = False
            if len(module.attention_blocks) > 1:
                module.attention_blocks[1].mode = 'OFF'
                module.attention_blocks[1].q_bank_enabled = False
                module.attention_blocks[1].k_bank_enabled = False
                module.attention_blocks[1].v_bank_enabled = False
                module.attention_blocks[1].norm_bank_enabled = False
            if config is not None and config.outputs[i]  and mode.name != 'OFF':
                module.attention_blocks[0].mode = mode.name
                module.attention_blocks[0].q_bank_enabled = config.q_bank_enabled
                module.attention_blocks[0].k_bank_enabled = config.k_bank_enabled
                module.attention_blocks[0].v_bank_enabled = config.v_bank_enabled
                module.attention_blocks[0].norm_bank_enabled = config.norm_bank_enabled
                if len(module.attention_blocks) > 1:
                    module.attention_blocks[1].mode = mode.name
                    module.attention_blocks[1].q_bank_enabled = config.q_bank_enabled
                    module.attention_blocks[1].k_bank_enabled = config.k_bank_enabled
                    module.attention_blocks[1].v_bank_enabled = config.v_bank_enabled
                    module.attention_blocks[1].norm_bank_enabled = config.norm_bank_enabled
                if len(module.attention_blocks) > 2:
                    print('whoa')
            i += 1

    def clear_modules(self):
        for module in self.modules:
            module.clean()
        self.set_mode(RefMode.OFF)
        self.set_kv_mode(RefMode.OFF, len(self.modules))
        self.set_normal_mode(RefMode.OFF)
        self.set_motion_mode(None, RefMode.OFF)
        motion_modules = list(filter(is_named_module_transformer_block, self.model.named_modules()))
        for _, module in motion_modules:
            for i in range(2):
                module.attention_blocks[i].q_bank = None
                module.attention_blocks[i].k_bank = None
                module.attention_blocks[i].v_bank = None
                module.attention_blocks[i].norm_bank = None
                module.attention_blocks[i].q_bank_enabled = False
                module.attention_blocks[i].k_bank_enabled = False
                module.attention_blocks[i].v_bank_enabled = False
                module.attention_blocks[i].norm_bank_enabled = False

    def set_cfg(self, cfg: bool):
        for module in self.modules:
            module.is_cfg = cfg

    def set_attention_bound(self, attention_bound: float):
        for module in self.modules:
            module.attention_bound = attention_bound

    def set_content_fidelity(self, content_fidelity: float): 
        for module in self.modules:
            module.content_fidelity = content_fidelity

    def set_normal_add(self, normal_add: bool): 
        for module in self.modules:
            module.normal_add = normal_add

    def set_spatial_query(self, spatial_query: bool):
        for module in self.modules:
            module.spatial_query = spatial_query
