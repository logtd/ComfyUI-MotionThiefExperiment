
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
