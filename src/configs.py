import random
from annotations import get_annotation_layer_keys
class _Config(object):
    seed = random.randint(0,2**32)
    epochs = 10
    max_length = None
    batch_size = 20
    learning_rate = 5e-5
    dropout = 0.5
    freeze_layers = None 
    num_labels = None
    longformer = False
    classifier_second_layer = None
    min_label_count_threshold = 15


def get_config(layer,context,seed=None):
    config = _Config()
    print(layer,context)
    config.num_labels = len(get_annotation_layer_keys(layer))

    if seed is not None:
        config.seed = seed
    if layer == "1":
        config.num_labels = 2        
        config.epochs = 40
        config.classifier_second_layer = 256
    elif layer == "2":
        config.epochs = 60
        config.classifier_second_layer = 1024
    elif layer == "3":
        config.epochs = 60
        config.classifier_second_layer = 1024
    elif layer == "4":
        config.epochs = 60
        config.classifier_second_layer = 1024


    return config


    