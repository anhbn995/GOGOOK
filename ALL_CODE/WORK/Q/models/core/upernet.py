import math
import tensorflow as tf

from models.layers.activation import GELU, Snake
# from tensorflow_addons.layers import AdaptiveAveragePooling2D
from models.layers.swin_transform import SwinTransformerModel
from tensorflow.keras.layers import Conv2D, BatchNormalization, Concatenate, Dropout, LayerNormalization, \
                                    ReLU, LeakyReLU, PReLU, ELU

CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}

def split_col(array, num_cols, num_take):
  list_split = []
  start = 0
  for i in range(num_cols):
    end = start + num_take
    if end < array.shape[1]:
      list_split.append(array[:,start:end,:,:])
    else:
      list_split.append(array[:,array.shape[1]-num_take:array.shape[1],:,:])
    start += num_take
  return list_split

def split_row(array, num_rows, num_take):
  list_split = []
  start = 0
  for i in range(num_rows):
    end = start + num_take
    if end < array.shape[-2]:
      list_split.append(array[:,:,:,start:end,:])
    else:
      list_split.append(array[:,:,:,array.shape[-2]-num_take:array.shape[-2],:])
    start += num_take
  return list_split

def adaptive_avg_pool2d(x5, out_size=2, input_shape=7):
  split_cols = split_col(x5, out_size, math.ceil(input_shape/out_size))
  split_cols = tf.stack(split_cols, axis=1)
  split_rows = split_row(split_cols, out_size, math.ceil(input_shape/out_size))
  split_rows = tf.stack(split_rows, axis=3)
  out_vect = tf.reduce_mean(split_rows, axis=[2, 4])
#   print(out_vect.shape)
  return out_vect

class Uperhead():
    def __init__(self, ppm_filter=128, ppm_scales=(1, 2, 3, 6), num_class=1, use_bias=False):
        self.use_bias = use_bias
        self.conv_filter = ppm_filter
        self.ppm_scales = ppm_scales
        self.num_class = num_class

    def _PPM(self, inputs):
        list_ppm_out = []
        for i in range(len( self.ppm_scales)):
            size = self.ppm_scales[i]
            # x = AdaptiveAveragePooling2D(output_size=size, name='uper_ppm_avgpool_%s'%str(i+1))(inputs)
            x = adaptive_avg_pool2d(x5=inputs, out_size=size, input_shape=inputs.shape[1])
            x = Conv2D(filters=self.conv_filter, kernel_size=(1,1), strides=(1,1), use_bias=self.use_bias,
                        name='uper_ppm_conv_%s'%str(i+1))(x)
            x = BatchNormalization(momentum=0.1, epsilon=1e-5, name='uper_ppm_batchnorm_%s'%str(i+1))(x)
            x = ReLU(name='uper_ppm_relu_%s'%str(i+1))(x)
            x = tf.image.resize(x, size=(inputs.shape[1], inputs.shape[2]))
            list_ppm_out.append(x)
        return list_ppm_out
    
    def _bottleneck(self, inputs, name):
        x = Conv2D(filters=self.conv_filter, kernel_size=(3,3), strides=(1,1), use_bias=self.use_bias,
                    padding='same', name='uper_bottleneck_conv_%s'%(name))(inputs)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, name='uper_bottleneck_batchnorm_%s'%(name))(x)
        x = ReLU(name='uper_bottleneck_relu_%s'%(name))(x)
        return x

    def _Fpn_in(self, inputs):
        list_fpnin = []
        for i in reversed(range(len(inputs)-1)):
            x = Conv2D(filters=self.conv_filter, kernel_size=(1,1), strides=(1,1), use_bias=self.use_bias,
                        name='uper_fpnin_conv_%s'%str(i+1))(inputs[i])
            x = BatchNormalization(momentum=0.1, epsilon=1e-5, name='uper_fpnin_batchnorm_%s'%str(i+1))(x)
            x = ReLU(name='uper_fpnin_relu_%s'%str(i+1))(x)
            list_fpnin.append(x)
        return list_fpnin
    
    def _Fpn_out(self, inputs):
        list_fpnout = []
        for i in range(len(inputs)):
            x = Conv2D(filters=self.conv_filter, kernel_size=(3,3), strides=(1,1), use_bias=self.use_bias,
                        padding='same', name='uper_fpnout_conv_%s'%str(i+1))(inputs[i])
            x = BatchNormalization(momentum=0.1, epsilon=1e-5, name='uper_fpnout_batchnorm_%s'%str(i+1))(x)
            x = ReLU(name='uper_fpnout_relu_%s'%str(i+1))(x)
            list_fpnout.append(x)
        return list_fpnout

    def __call__(self, inputs):
        ppm_in = inputs[-1]
        ppm_out = self._PPM(inputs=ppm_in)
        list_bottleneck = [ppm_in]
        list_bottleneck.extend(ppm_out)
        bottleneck_in = Concatenate(axis=-1)(list_bottleneck)
        bottleneck_out = self._bottleneck(bottleneck_in, name='ppm')
        fpn_features = [bottleneck_out]

        fpnin_out = self._Fpn_in(inputs=inputs)
        list_fpnout_in = []
        for i in range(len(fpnin_out)):
            h, w = fpnin_out[i].shape[1], fpnin_out[i].shape[2]
            # f_resize = Resizing(height=h, width=w, interpolation='bilinear', crop_to_aspect_ratio=False)(bottleneck_out)
            f_resize = tf.image.resize(bottleneck_out, size=(h,w))
            fpnout_in = fpnin_out[i] + f_resize
            list_fpnout_in.append(fpnout_in)
        fpnout_out = self._Fpn_out(list_fpnout_in)
        fpn_features.extend(fpnout_out)
       
        fpn_features.reverse()
        for i in range(1, len(inputs)):
            h, w = fpn_features[0].shape[1], fpn_features[0].shape[2]
            # fpn_features[i] = Resizing(height=h, width=w, interpolation='bilinear', crop_to_aspect_ratio=False)(fpn_features[i])
            fpn_features[i] = tf.image.resize(fpn_features[i], size=(h,w))

        output = Concatenate(axis=-1)(fpn_features)
        output = self._bottleneck(output, name='out')
        output = Dropout(rate=0.1)(output)
        output = Conv2D(filters=self.num_class, kernel_size=(1,1), name='uper_final_conv', use_bias=True)(output)    
        
        return output

class FCNhead():
    def __init__(self, conv_filter=128, num_class=1):
        self.conv_filter = conv_filter
        self.num_class = num_class

    def __call__(self, inputs):
        x = tf.image.resize(inputs[2], size=(inputs[0].shape[1], inputs[0].shape[2]))
        x = Conv2D(filters=self.conv_filter, kernel_size=(1,1), name='fcn_module_conv')(x)
        x = BatchNormalization(name='fcn_module_batchnorm')(x)
        x = ReLU(name='fcn_module_relu')(x)
        x = Dropout(0.1)(x)
        output = Conv2D(filters=self.num_class, kernel_size=(1,1), name='fcn_aux_conv', use_bias=True)(x)
        
        return output

def upernet(input_shape=(256,256,4), model_name='swin_tiny_224', include_top=False, patch_size=(4, 4), 
          mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
          norm_layer='LayerNormalization', ape=False, patch_norm=True,use_checkpoint=False, ppm_filter=128, 
          ppm_scales=(1, 2, 3, 6), n_labels=1, use_bias=False, conv_filter=128, activation_fn='Sigmoid', name='upernet'):

  # activation_func = eval(activation_fn)
  activation_fn = activation_fn.lower()
  inputs = tf.keras.Input(input_shape)
  in_chans = input_shape[-1]
  img_size = (input_shape[0], input_shape[1])
  norm_layer = eval(norm_layer)
  if model_name in CFGS:
    params = CFGS[model_name]
    depths = params['depths']
    num_heads = params['num_heads']
    embed_dim = params['embed_dim']
    window_size = params['window_size']
    
  out_backbones = SwinTransformerModel(model_name=model_name, include_top=include_top, img_size=img_size, patch_size=patch_size, 
                                      in_chans=in_chans, num_classes=n_labels,embed_dim=embed_dim, depths=depths, num_heads=num_heads, 
                                      window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, ape=ape, patch_norm=patch_norm, use_checkpoint=use_checkpoint)(inputs)
  outputs = Uperhead(ppm_filter=ppm_filter, ppm_scales=ppm_scales, num_class=n_labels, use_bias=use_bias)(out_backbones)

  height, width = img_size
  outputs = tf.keras.layers.Resizing(height, width, interpolation='nearest')(outputs)
  outputs = tf.keras.layers.Activation(activation=activation_fn, name='final_act_output')(outputs)

  out_aux = FCNhead(conv_filter=conv_filter, num_class=n_labels)(out_backbones)
  out_aux = tf.keras.layers.Resizing(height, width, interpolation='nearest')(out_aux)
  out_aux = tf.keras.layers.Activation(activation=activation_fn, name='final_aux_output')(out_aux)
  model = tf.keras.Model(inputs=inputs, outputs=[outputs, out_aux])
  return model

if __name__=="__main__":
    inputs = tf.keras.Input((256,256,4))
    out_backbones = SwinTransformerModel(img_size=(256,256), in_chans=4, embed_dim=128)(inputs)
    outputs = Uperhead(num_class=1)(out_backbones)
    out_aux = FCNhead(conv_filter=64, num_class=1)(out_backbones)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, out_aux])
    model.summary()