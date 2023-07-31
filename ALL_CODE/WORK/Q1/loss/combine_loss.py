import tensorflow as tf
from loss.iou_box_seg_loss import iou_seg
from loss.tversky_loss import focal_tversky
from loss.ms_ssim_loss import ms_ssim

def hybrid_loss(y_true, y_pred):

    loss_focal = focal_tversky(y_true, y_pred, alpha=0.25, gamma=2)
    loss_iou = iou_seg(y_true, y_pred)
    
    # (x) 
    # loss_ssim = ms_ssim(y_true, y_pred, max_val=1.0, filter_size=4)
    
    return loss_focal + loss_iou #+ loss_ssim

# @tf.function
# def segmentation_loss(y_true, y_pred, weight=None, N_CLASSES=4):
#     cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#     cross_entropy_loss = cce(y_true=y_true, y_pred=y_pred)
#     dice_loss = gen_dice(y_true, y_pred, N_CLASSES)
#     return 0.5 * cross_entropy_loss + 0.5 * dice_loss

@tf.function
def segmentation_loss(y_true, y_pred, weight=None, N_CLASSES=1):
    if N_CLASSES == 1:
        cce = tf.keras.losses.BinaryCrossentropy()
    else:
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    cross_entropy_loss = cce(y_true=y_true, y_pred=y_pred)
#     dice_loss = gen_dice(y_true, y_pred, N_CLASSES)
    loss_iou = iou_seg(y_true, y_pred)
    return cross_entropy_loss

@tf.function
def dice_per_class(y_true, y_pred, eps=1e-5):
    intersect = tf.reduce_sum(y_true * y_pred)
    y_sum = tf.reduce_sum(y_true * y_true)
    z_sum = tf.reduce_sum(y_pred * y_pred)
    loss = 1 - (2 * intersect + eps) / (z_sum + y_sum + eps)
    return loss

@tf.function
def gen_dice(y_true, y_pred, weight=None, N_CLASSES=2):
    """both tensors are [b, h, w, classes] and y_pred is in logit form"""
    # [b, h, w, classes]
    pred_tensor = tf.nn.softmax(y_pred)
    if weight is None:
            weight = [1] * N_CLASSES
    # assert pred_tensor.shape == y_true.shape, 'predict {} & target {} shape do not match'.format(pred_tensor.shape, y_true.shape)
    loss = 0.0
    for c in range(N_CLASSES):
        loss += dice_per_class(y_true[:, :, :, c], pred_tensor[:, :, :, c])*weight
    return loss/N_CLASSES

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def balanced_cross_entropy_loss(y_true, y_pred):
  _epsilon = _to_tensor(tf.keras.backend.epsilon(), y_pred.dtype )
  y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
  y_pred = tf.math.log(y_pred/ (1 - y_pred))

  y_true = tf.cast(y_true, tf.float32)
  count_neg = tf.reduce_sum(input_tensor=1. - y_true)
  count_pos = tf.reduce_sum(input_tensor=y_true)
  beta = count_neg / (count_neg + count_pos)
  pos_weight = beta / (1 - beta)
  cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
  cost = tf.reduce_mean(input_tensor=cost * (1 - beta))
  return tf.compat.v1.where(tf.equal(count_pos, 0.0), 0.0, cost)