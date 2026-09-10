from models.smart_transformer import SMART
from models.smart_transformer import SMART_Base
from models.smart_transformer import SMART_Base_22K

__factory = {
    'SMART': SMART,
    'SMART_Base': SMART_Base,
    'SMART_Base_22K': SMART_Base_22K
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)