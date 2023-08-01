import tensorflow as tf



bce = tf.keras.losses.BinaryCrossentropy()
def pre_process_binary_cross_entropy(label, inputs):
    # preprocess data
    y = label
    loss = 0
    w_loss=1.0
    for tmp_p in inputs:
        tmp_y = tf.cast(y, dtype=tf.float32)
        mask = tf.dtypes.cast(tmp_y > 0., tf.float32)
        b,h,w,c=mask.get_shape()
        positives = tf.math.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
        negatives = h*w*c-positives

        beta2 = positives / (negatives + positives) # negatives in hed
        beta = negatives / (positives + negatives) # positives in hed
        pos_w = tf.where(tf.equal(y, 0.0), beta2, beta)
        
        l_cost = bce(y_true=tmp_y, y_pred=tmp_p, sample_weight=pos_w)
        loss += (l_cost*w_loss)
    return loss

def binary_cross_entropy(label, inputs):
    # preprocess data
    y = label
    loss = 0
    w_loss=1.0
    for tmp_p in inputs:
        tmp_y = tf.cast(y, dtype=tf.float32)
        
        l_cost = bce(y_true=tmp_y, y_pred=tmp_p)
        loss += (l_cost*w_loss)
    return loss