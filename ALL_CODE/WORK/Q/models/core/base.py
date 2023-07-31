import json
import warnings

from tensorflow.keras.applications import *
from tensorflow.keras.models import Model

# layer_cadidates = json.load(open("/home/quyet/DATA_ML/WorkSpace/segmentation/models/core/layer_cadidates.json"))
layer_cadidates = json.load(open("/home/skm/SKM/WORK/ALL_CODE/WORK/Q/models/core/layer_cadidates.json"))

def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model
    
    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization    
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model

def bach_norm_checker(backbone_name, batch_norm):
    '''batch norm checker'''
    if 'VGG' in backbone_name:
        batch_norm_backbone = False
    else:
        batch_norm_backbone = True
        
    if batch_norm_backbone != batch_norm:       
        if batch_norm_backbone:    
            param_mismatch = "\n\nBackbone {} uses batch norm, but other layers received batch_norm={}".format(backbone_name, batch_norm)
        else:
            param_mismatch = "\n\nBackbone {} does not use batch norm, but other layers received batch_norm={}".format(backbone_name, batch_norm)
            
        warnings.warn(param_mismatch);
        
def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm):
    '''
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`
    
    Input
    ----------
        backbone_name: the bakcbone model name. Expected as one of the `tensorflow.keras.applications` class.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0,7]
                       
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        input_tensor: the input tensor 
        depth: number of encoded feature maps. 
               If four dwonsampling levels are needed, then depth=4.
        
        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        model: a keras backbone model.
        
    '''
    
    cadidate = layer_cadidates[backbone_name]
    
    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max
    # ----- #
    
    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None,)
    
    X_skip = []
    
    for i in range(depth):
        X_skip.append(backbone_.get_layer(cadidate[i]).output)
        
    model = Model(inputs=[input_tensor,], outputs=X_skip, name='{}_backbone'.format(backbone_name))
    
    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)
    
    return model