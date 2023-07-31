import os
import numpy as np
import tensorflow as tf


class SavebestweightsandEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=5, weights_path='./weights/model.h5', restore=False):
        super(SavebestweightsandEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None
        if not os.path.exists(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))
        self.weights_path_val = weights_path.replace('.h5','_val.h5')
        self.weights_path_train = weights_path.replace('.h5','_train.h5')
        self.restore = restore

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait_val = 0
        self.wait_train = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_val = np.Inf
        self.best_train = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get("val_loss")
        current_train = logs.get("loss")
        
        if np.less(current_train, self.best_train):
            self.best_train = current_train
            self.wait_train = 0
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_path_train)
            print("\nSave best train weights.")
        else:
            self.wait_train += 1
        
        if np.less(current_val, self.best_val):
            self.best_val = current_val
            self.wait_val = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_path_val)
            print("Save best val weights.")
        else:
            self.wait_val += 1
            if self.wait_val >= self.patience:
                print("Val loss doesn't improve.")
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore:
                    print("Restoring model weights from the end of the best epoch.")
                    self.model.set_weights(self.weights_path_val)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))