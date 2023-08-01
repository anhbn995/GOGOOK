import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import unet_basic
from utils import get_data
from tensorflow.keras.callbacks import TensorBoard

#input : trainning dataset, model 
#output: weight model

def training(data_dir, model_path, size_model, fp_out_model, **kwargs):
    model = unet_basic((size_model,size_model,4))
    print(model.summary())

    path_train = data_dir
    X, y = get_data(path_train, size=size_model, train=True, uint8_type=True)
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

    tensorboard_callback = TensorBoard(log_dir=path/to/log-directory, histogram_freq=1)

    callbacks = [
        EarlyStopping(patience=patience_early, verbose=verbose),
        ReduceLROnPlateau(factor=factor, patience=patience_reduce, min_lr=min_lr, verbose=verbose),
        ModelCheckpoint(fp_out_model, verbose=verbose, save_best_only=True, save_weights_only=True)
    ]
    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=int(epochs), callbacks=callbacks,
                                validation_data=(X_valid, y_valid))
    
if __name__ == '__main__':

    from training_config import *
    


    
    mota_dulieu_train = f"Duong dan cua model: {fp_out_model} \n Data goc la: {data_dir} \n neu co pretrain: {model_path} \n"
    file_save_mota = fp_out_model.replace('.h5', '.txt')
    with open(file_save_mota, "w") as file:
        file.write(mota_dulieu_train)
    training(data_dir, model_path, size_model, fp_out_model)