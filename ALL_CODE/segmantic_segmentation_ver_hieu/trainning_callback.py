import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import unet_basic
from utils import get_data
#input : trainning dataset, model 
#output: weight model


import keras
import requests
class TrainingCallback(keras.callbacks.Callback):

    def __init__(self, task_id):
        super(TrainingCallback, self).__init__()
        self.task_id = task_id

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        payload = []
        for key in keys:
            payload.append({
                'key': key,
                'value': str(logs[key])
            })
        payload = {
            'task_id': self.task_id,
            'epoch': epoch + 1,
            'payload': payload
        }
        # url = '{}/internal/training/report'.format(HOSTED_ENDPOINT)
        # print(url)
        # print(payload)
        # request = requests.post(url, json=payload)
        # print(request)


def training(data_dir, model_path, size_model, fp_out_model, **kwargs):

    model = unet_basic((size_model,size_model,4))
    print(model.summary())

    path_train = data_dir
    X, y = get_data(path_train, size=size_model, train=True)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2019)

    if not os.path.exists(model_path):
        pass
    else:
        model.load_weights(model_path)

    patience_early = kwargs.get('patience_early') or 10
    factor = kwargs.get('factor') or 0.1
    patience_reduce=kwargs.get('patience_reduce') or 3
    min_lr = kwargs.get('min_lr') or 0.00001
    verbose= 1
    epochs = 100
    batch_size = 10

    callbacks = [
        EarlyStopping(patience=patience_early, verbose=verbose),
        ReduceLROnPlateau(factor=factor, patience=patience_reduce, min_lr=min_lr, verbose=verbose),
        ModelCheckpoint(fp_out_model, verbose=verbose, save_best_only=True, save_weights_only=True)
    ]
    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=int(epochs), callbacks=callbacks,
                                validation_data=(X_valid, y_valid))
    
if __name__ == '__main__':
    data_dir = r'/home/skm/SKM16/Tmp/HIEEU/training_dataset'
    model_path = ''
    fp_out_model = r'/home/skm/SKM16/X/Data_MachineLearning/V2_sentinel/training_dataset/model/building_v2.h5'
    size_model = 256
    training(data_dir, model_path, size_model, fp_out_model)