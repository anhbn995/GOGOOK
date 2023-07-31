import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_image_ops

def convert_image_dtype(image, dtype, saturate=False, name=None):
  image = ops.convert_to_tensor(image, name='image')
  dtype = dtypes.as_dtype(dtype)
  if not dtype.is_floating and not dtype.is_integer:
    raise AttributeError('dtype must be either floating point or integer')
  if dtype == image.dtype:
    return array_ops.identity(image, name=name)

  with ops.name_scope(name, 'convert_image', [image]) as name:
    if image.dtype.is_integer and dtype.is_integer:
      scale_in = image.dtype.max
      scale_out = dtype.max
      if scale_in > scale_out:
        scale = (scale_in + 1) // (scale_out + 1)
        scaled = math_ops.floordiv(image, scale)

        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)
      else:
        if saturate:
          cast = math_ops.saturate_cast(image, dtype)
        else:
          cast = math_ops.cast(image, dtype)
        scale = (scale_out + 1) // (scale_in + 1)
        return math_ops.multiply(cast, scale, name=name)
    elif image.dtype.is_floating and dtype.is_floating:
      return math_ops.cast(image, dtype, name=name)
    else:
      if image.dtype.is_integer:
        cast = math_ops.cast(image, dtype)
        scale = 1. / image.dtype.max
        return math_ops.multiply(cast, scale, name=name)
      else:
        scale = dtype.max + 0.5
        scaled = math_ops.multiply(image, scale)
        if saturate:
          return math_ops.saturate_cast(scaled, dtype, name=name)
        else:
          return math_ops.cast(scaled, dtype, name=name)

def adjust_brightness(image, delta):
  with ops.name_scope(None, 'adjust_brightness', [image, delta]) as name:
    image = ops.convert_to_tensor(image, name='image')
    orig_dtype = image.dtype

    if orig_dtype in [dtypes.float16, dtypes.float32]:
      flt_image = image
    else:
      flt_image = convert_image_dtype(image, dtypes.float32)

    adjusted = math_ops.add(
        flt_image, math_ops.cast(delta, flt_image.dtype), name=name)
    return convert_image_dtype(adjusted, orig_dtype, saturate=True)

def random_brightness(image, max_delta, min_delta, seed=None):
    if max_delta < 0 or min_delta <0:
        raise ValueError('max_delta and min_delta must be non-negative.')

    delta = random_ops.random_uniform([], min_delta, max_delta, seed=seed)
    return adjust_brightness(image, delta)

def adjust_hue(image, delta, name=None):
    with ops.name_scope(name, 'adjust_hue', [image]) as name:
        if context.executing_eagerly():
            if delta < -1 or delta > 1:
                raise ValueError('delta must be in the interval [-1, 1]')
        image = ops.convert_to_tensor(image, name='image')
        orig_dtype = image.dtype
        if orig_dtype in (dtypes.float16, dtypes.float32):
            flt_image = image
        else:
            flt_image = convert_image_dtype(image, dtypes.float32)

        rgb_altered = gen_image_ops.adjust_hue(flt_image, delta)
        return convert_image_dtype(rgb_altered, orig_dtype)

def random_hue(image, max_delta, min_delta ,seed=None):
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')

    if max_delta < 0 or min_delta <0:
        raise ValueError('max_delta and min_delta must be non-negative.')

    delta = random_ops.random_uniform([], min_delta, max_delta, seed=seed)
    return adjust_hue(image, delta)

def random_rotate(image, label, rotation=20):
    rot = np.random.uniform(-rotation*np.pi/180, rotation*np.pi/180)
    modified = tfa.image.rotate(image, rot)
    m_label = tfa.image.rotate(label, rot)
    return modified, m_label

def random_rot_flip(image, label, width, height):
    m_label = tf.reshape(label, (width, height, 1))
    axis = np.random.randint(0, 2)
    if axis == 1:
        # vertical flip
        modified = tf.image.flip_left_right(image=image)
        m_label = tf.image.flip_left_right(image=m_label)
    else:
        # horizontal flip
        modified = tf.image.flip_up_down(image=image)
        m_label = tf.image.flip_up_down(image=m_label)
    # rot 90
    k_90 = np.random.randint(4)
    modified = tf.image.rot90(image=modified, k=k_90)
    m_label = tf.image.rot90(image=m_label, k=k_90)

    m_label = tf.reshape(m_label, (width, height))
    return modified, m_label

def data_augment(image, label, N_CLASSES):
    rand1, rand2, rand3, rand4, rand5 = np.random.uniform(size=(5, 1))
    w,h = image.shape[0], image.shape[1]
    status = False
    if rand1 > 0.25:
        status = True
        modified, m_label = random_rot_flip(image, label, w, h)
    elif rand2 > 0.25:
        status = True
        try: 
            modified, m_label = random_rotate(image, label, rotation=20)
        except:
            modified, m_label = random_rotate(image, label, rotation=20)
    elif rand3 > 0.25:
        status = True
        try:
            modified  = tf.image.random_hue(image, max_delta=0.5)
            m_label = m_label
        except:
            modified  = tf.image.random_hue(image, max_delta=0.5)
            m_label = label
    elif rand4 > 0.25:
        status = True
        try:
            modified = tf.image.random_brightness(image, max_delta=0.2)
            m_label = m_label
        except:
            modified = tf.image.random_brightness(image, max_delta=0.2)
            m_label = label
            
    if not status:
        modified, m_label = image, label
    m_label = tf.cast(m_label, tf.uint8)
    m_label = tf.one_hot(m_label, depth=N_CLASSES)
    return modified, m_label