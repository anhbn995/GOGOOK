import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob, shutil, warnings, os
from utils.color_image import Color_image
from utils.augmentations import Augmentations
from multiprocessing import Pool
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus, \
                        weighted_cross_entropy_loss, pre_process_binary_cross_entropy, binary_cross_entropy, IoULoss
                        
from tensorflow.compat.v1.keras.backend import set_session

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))


class DataParser():
    def __init__(self, annotation, batch_size, augmentations=None, color_image=None):
        self.total_data = annotation
        self.batch_size = batch_size
        self.len_data = int(len(self.total_data)//self.batch_size)
        self.check_batch = self.len_data * self.batch_size
        self.augmentations = augmentations
        self.color_image = color_image
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.check_batch:
            filename = self.total_data[self.num: self.num+self.batch_size]
            # with Pool() as p:
            #     p.map(self.get_batch, filename)
            image, label = self.get_batch(filename)
            self.num += self.batch_size
            return image, label
        else:
            self.num = 0
            np.random.shuffle(self.total_data)
            raise StopIteration


    def get_batch(self, batch):
        images = []
        edgemaps = []
        for img_list in batch:
            im = np.load(img_list).transpose(1,2,0)#[...,:3]
            em = np.load(img_list.replace('npy_i', 'npy_m')).transpose(1,2,0)
            # print(im.shape, em.shape,"truoc")
            if self.augmentations != None:
                im, em = self.augmentations(im, em)
            if self.color_image != None:
                im = self.color_image(im)
                
            im = np.array(im/255., dtype=np.float32)
            em = em.astype(np.float32)

            images.append(im)
            edgemaps.append(em)
            # print(im.shape, em.shape)

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)
        return images, edgemaps
    
    def __len__(self):
        return self.len_data


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred = my_model(image_data, training=True)
        loss = binary_cross_entropy(target, pred)

    gradients = tape.gradient(loss, my_model.trainable_variables)
    del tape
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    global_steps.assign_add(1)
    with writer.as_default():
        tf.summary.scalar("loss/loss", loss, step=global_steps)
    writer.flush()
        
    return loss.numpy()

def val_step(image_data, target):
    pred = my_model(image_data, training=False)
    loss = binary_cross_entropy(target, pred)
    return loss.numpy()

if __name__ == '__main__':
    import time
    import datetime
    x = time.time()
    VisEff = Color_image()
    Augmen = Augmentations()
    
    "cai dau tien"
    # my_model = Model_U2Net(256, 3)
    # my_model.load_weights('/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/U2net/model/model_u2net.h5')

    "cai thu hai "
    # my_model = DexiNed(256, 3)

    "cai thu ba"
    # my_model = Adalsn(256, 3)
    
    "cai thu ba"
    # my_model = Model_UNet3plus(256, 3)


    "sat lo"
    # my_model = Model_U2Net(128, 4)
    # my_model = Model_UNet3plus(128, 4)
    # my_model = DexiNed(128, 4)
    


    # my_model = DexiNed(480, 3)
    # my_model = Model_U2Net(256, 3)
    # my_model = Model_UNet3plus(480, 3)
    # my_model = Adalsn(480, 3)
    # my_model.load_weights('/mnt/data/Nam_work_space/model/adalsn_farm_v3.h5')
    # my_model.load_weights('/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/U2net/model/model_u2net.h5')

    # path1 = glob.glob('/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/data_train/img_256/*.npy')
    # np.random.shuffle(path1)
    # path1 = glob.glob('/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/U2net/data_train_bo_dagood/npy_i_256/*.npy')
    # np.random.shuffle(path1)
    
    # path1 = glob.glob('/home/skm/SKM16/IMAGE/ZZ_ZZ/cloud_shadow/U2net/data_train_bo_dagood/npy_i_256/*.npy')
    # np.random.shuffle(path1)
    

    
    # path2 = glob.glob('/mnt/data/Nam_work_space/data_train/image_40/*.npy')
    # np.random.shuffle(path2)
    # path3 = glob.glob('/mnt/data/Nam_work_space/data_train/image_train/*.npy')
    # np.random.shuffle(path3)
    # path4 = glob.glob('/mnt/data/Nam_work_space/data_train/image_update/*.npy')
    # np.random.shuffle(path4)

    # path3 = glob.glob('/mnt/data/Nam_work_space/model_malay/image_update/*.npy')
    # np.random.shuffle(path3)
    
    """
    path1 = glob.glob('/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Gen_for_u2net/gen_numpy_cut128_uint8/npy_i_256/*.npy')
    np.random.shuffle(path1)
    # TRAIN_LOGDIR = '/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/U2net/data_train_bo_dagood/model_256_khac_loss_50/logs'
    TRAIN_LOGDIR = '/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Gen_for_u2net/DexiNed/logs'
    """
    
    
    import os
    import datetime
    now = datetime.datetime.now()
    dt = now.strftime("%Y%m%d_%H%M%S")
    
    
    size_model = 32
    my_model = Model_U2Net(size_model, 4)
    name_model = f"Model_U2Net_size{size_model}" + '_' + dt + ".h5"
    dir_dataset = f"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/NPY_FOR_U2net_3net/size_{size_model}/npy_i"
    DIR_MODEL = f'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard/U2Net/model_{size_model}'
    TRAIN_LOGDIR = os.path.join(DIR_MODEL,'logs')
    FP_MODEL = os.path.join(DIR_MODEL, name_model)
    TRAIN_EPOCHS = 50
    
    best_val_loss = 1
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

    
    path1 = glob.glob(os.path.join(dir_dataset,'*.npy'))
    # print("z"*100, "\n", path1)
    np.random.shuffle(path1)
    
    
    alpha = 0.8
    # data_train = 5*path1[:int(len(path1)*alpha)]+11*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]+path4[:int(len(path4)*alpha)]
    # data_val = 5*path1[int(len(path1)*alpha):]+11*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]+path4[int(len(path4)*alpha):]  
    data_train = path1[:int(len(path1)*alpha)]
    # print(data_train)
    data_val = path1[int(len(path1)*alpha):]
    np.random.shuffle(data_train)
    np.random.shuffle(data_val)
    batch_size = 10
    traindata = DataParser(data_train, batch_size, Augmen)#, VisEff)
    valdata = DataParser(data_val, batch_size)
    len_train = len(traindata)
    len_val = len(valdata)
    # print("z"*100, "\n", len_train, len_val)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    if os.path.exists(TRAIN_LOGDIR): 
        try: shutil.rmtree(TRAIN_LOGDIR)
        except: pass
        
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(TRAIN_EPOCHS):
        print('Epoch: %s / %s'%(str(epoch+1),str(TRAIN_EPOCHS)))
        with tqdm(total=len_train,desc=f'Train',postfix=dict,mininterval=0.3) as pbar:
            total_train = 0
            for image_data, target in traindata:
                # print(image_data.shape)
                results = train_step(image_data, target)
                total_train += results
                pbar.set_postfix(**{'total_loss' : results})
                pbar.update(1)              
            pbar.set_postfix(**{'total_train' : total_train/len_train})
            pbar.update(1)    
                 
        with tqdm(total=len_val,desc=f'Val',postfix=dict,mininterval=0.3) as pbar:
            total_val = 0
            for image_data, target in valdata:
                results = val_step(image_data, target)
                total_val += results
                pbar.set_postfix(**{'total_val' : results})
                pbar.update(1)
            pbar.set_postfix(**{'total_val' : total_val/len_val})
            pbar.update(1)
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val/len_val, step=epoch)
            validate_writer.flush()
            
            if best_val_loss>=total_val/len_val:
                # my_model.save_weights(os.path.join("/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/U2net/data_train_bo_dagood/model_256_good_boda", f"model_u2net_256_boda_good.h5"))
                # my_model.save_weights(os.path.join("/home/skm/SKM_OLD/ZZ_ZZ/cloud_shadow/U2net/data_train_bo_dagood/model_256_khac_loss_50", f"model_U2net_256_boda_good_binarycross_entropy_50.h5"))
                my_model.save_weights(FP_MODEL)
                best_val_loss = total_val/len_val

        print(44*'-'+21*'*'+44*'-')
        print() 
    y = time.time()
    delta = y - x
    delta_time = datetime.timedelta(seconds=delta)
    time_str = str(delta_time)

    mota_dulieu_train = f"Duong dan cua model: {FP_MODEL} \n Data goc la: {dir_dataset} \n neu co pretrain: None \n het bao lau: {time_str}\n ok"
    file_save_mota = FP_MODEL.replace('.h5', '.txt')
    print(file_save_mota)
    with open(file_save_mota, "w") as file:
        file.write(mota_dulieu_train)
    print(time_str) 