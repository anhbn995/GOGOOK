import tensorflow as tf
import matplotlib.pyplot as plt

class Displayresultsafterepoch(tf.keras.callbacks.Callback):
    def __init__(self, img_path):
        super(Displayresultsafterepoch, self).__init__()
        self.img_path = img_path
        
    def on_train_begin(self, logs=None):
        self.img_1 = rasterio.open(self.img_path).read().swapaxes(0,1).swapaxes(1,2)
        
    def on_train_end(self, logs=None):
        print("Tai sao ko hien thi.")
        self.img_1 = rasterio.open(self.img_path).read().swapaxes(0,1).swapaxes(1,2)
        self.img_2 = self.model.predict(tf.expand_dims(self.img_1, axis=0)) 
        self.img_2 = np.argmax(self.img_2, axis=-1)
        self.img_2 = self.img_2.astype('int16')[0]
        
        fig = plt.figure(figsize=(15, 7))
        rows = 1
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.img_1/255)
        fig.add_subplot(rows, columns, 2)
        plt.imshow(self.img_2)