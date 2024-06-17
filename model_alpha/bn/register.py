import torch

MODEL_CLASS_DICT = dict()
MODEL_FUNC_DICT = dict()

def register_model_class(model_class):
    MODEL_CLASS_DICT[model_class.__name__] = model_class
    return model_class

def register_model_func(model_func):
    MODEL_FUNC_DICT[model_func.__name__] = model_func
    return model_func