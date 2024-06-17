import torch

CLASS_DICT = dict()

def get_bottleneck_processor(class_name, *args, **kwargs):
    if class_name not in CLASS_DICT:
        print('ERROR CLASS!')
        return None

    get_class = CLASS_DICT[class_name](*args, **kwargs)
    return get_class