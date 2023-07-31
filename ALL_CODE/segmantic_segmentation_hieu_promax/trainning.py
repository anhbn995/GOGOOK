import os

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import unet_basic
from utils import get_data
# from tensorflow.keras.callbacks import TensorBoard
from training_config import *
 
 
import tensorflow as tf
from tensorflow import keras
#input : trainning dataset, model 
#output: weight model



def training(data_dir, model_path, size_model, fp_out_model, type_data, **kwargs):
    model = unet_basic((size_model,size_model,4))
    print(model.summary())

    path_train = data_dir
    X, y = get_data(path_train, size=size_model, train=True, uint8_type=type_data)
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
    epochs = 1000
    batch_size = 10

    tb_callback = keras.callbacks.TensorBoard(log_dir=log_tensorboard_dir, histogram_freq=1)

    # Tạo đối tượng writer để ghi thông tin vào TensorBoard
    writer = tf.summary.create_file_writer(log_tensorboard_dir)
    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with writer.as_default():
                tf.summary.scalar('accuracy_train', logs['accuracy'], step=epoch)

            # Lưu accuracy của tập validation vào TensorBoard
            with writer.as_default():
                tf.summary.scalar('accuracy_val', logs['val_accuracy'], step=epoch)
                
            # Lưu loss của tập train vào TensorBoard
            with writer.as_default():
                tf.summary.scalar('loss_train', logs['loss'], step=epoch)
            
            # Lưu loss của tập validation vào TensorBoard
            with writer.as_default():
                tf.summary.scalar('loss_val', logs['val_loss'], step=epoch)


    # Compile model với metric và optimizer tương ứng
    # callbacks = [tb_callback, CustomCallback()]
    callbacks = [
        EarlyStopping(patience=patience_early, verbose=verbose),
        ReduceLROnPlateau(factor=factor, patience=patience_reduce, min_lr=min_lr, verbose=verbose),
        ModelCheckpoint(fp_out_model, verbose=verbose, save_best_only=True, save_weights_only=True),
        tb_callback,
        CustomCallback()
    ]
    
    model.fit(X_train, y_train, batch_size=int(batch_size), epochs=int(epochs), callbacks=callbacks,
                                validation_data=(X_valid, y_valid))
    
if __name__ == '__main__':
    import time
    x = time.time()
    
    training(data_dir, model_path, size_model, fp_out_model, type_data)
    y = time.time()
    delta = y - x
    delta_time = datetime.timedelta(seconds=delta)
    time_str = str(delta_time)
    mota_dulieu_train = f"Duong dan cua model: {fp_out_model} \n Data goc la: {data_dir} \n neu co pretrain: {model_path} \n het bao lau: {time_str}\n ok"
    file_save_mota = fp_out_model.replace('.h5', '.txt')
    with open(file_save_mota, "w") as file:
        file.write(mota_dulieu_train)
    print(time_str)