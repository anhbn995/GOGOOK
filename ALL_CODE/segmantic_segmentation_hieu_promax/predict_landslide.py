import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import warnings
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import config
import time

x = time.time()
warnings.filterwarnings("ignore")
config_ = tf.compat.v1.ConfigProto()
config_.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config_))

def predict(model, fp_i, fp_P, md_s):
    with rasterio.open(fp_i) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0})
        h, w = raster.height, raster.width
        s_ = md_s - md_s //4
        p_ = int((md_s - s_) / 2)
        
        l_coor = []
        for s_y in range(0, h, s_):
            for s_x in range(0, w, s_): 
                x_ = s_x if s_x==0 else s_x - p_
                y_ = s_y if s_y==0 else s_y - p_
                    
                e_X = min(s_x + s_ + p_, w)
                e_Y = min(s_y + s_ + p_, h)
                
                x_c = e_X - x_
                y_c = e_Y - y_
                l_coor.append(tuple([x_, y_, x_c, y_c, s_x, s_y]))
                
        with tqdm(total=len(l_coor)) as pbar:
            with rasterio.open(fp_P,'w+', **meta, compress='lzw') as r:
                for x_, y_, x_c, y_c, s_x, s_y in l_coor:
                    img_d = raster.read(window=Window(x_, y_, x_c, y_c)).transpose(1,2,0)
                    m = np.pad(np.ones((s_, s_), dtype=np.uint8), ((p_, p_),(p_, p_)))
                    size_s = (s_, s_)
                    if y_c < md_s or x_c < md_s:
                        i_tem = np.zeros((md_s, md_s, img_d.shape[2]))
                        m = np.zeros((md_s, md_s), dtype=np.uint8)
                        if s_x == 0 and s_y == 0:
                            i_tem[(md_s - y_c):md_s, (md_s - x_c):md_s] = img_d
                            m[(md_s - y_c):md_s-p_, (md_s - x_c):md_s-p_] = 1
                            size_s = (y_c-p_, x_c-p_)
                        elif s_x == 0:
                            i_tem[0:y_c, (md_s - x_c):md_s] = img_d
                            if y_c == md_s:
                                m[p_:y_c-p_, (md_s - x_c):md_s-p_] = 1
                                size_s = (y_c-2*p_, x_c-p_)
                            else:
                                m[p_:y_c, (md_s - x_c):md_s-p_] = 1
                                size_s = (y_c-p_, x_c-p_)
                        elif s_y == 0:
                            i_tem[(md_s - y_c):md_s, 0:x_c] = img_d
                            if x_c == md_s:
                                m[(md_s - y_c):md_s-p_, p_:x_c-p_] = 1
                                size_s = (y_c-p_, x_c-2*p_)
                            else:
                                m[(md_s - y_c):md_s-p_, p_:x_c] = 1
                                size_s = (y_c-p_, x_c-p_)
                        else:
                            i_tem[0:y_c, 0:x_c] = img_d
                            m[p_:y_c, p_:x_c] = 1
                            size_s = (y_c-p_, x_c-p_)
                        img_d = i_tem
                    m = (m!=0)
            
                    if np.count_nonzero(img_d) > 0:
                        if len(np.unique(img_d)) <= 2:
                            pass
                        else:
                            #     y_P = model.predict(img_d[np.newaxis,...])
                            # except:
                            im
                            y_P = model.predict(img_d[np.newaxis,...]/255.)[0]
                            y_P = (y_P[0,...,0] > 0.5).astype(np.uint8)
                            y = y_P[m].reshape(size_s)
                            r.write(y[np.newaxis,...], window=Window(s_x, s_y, size_s[1], size_s[0]))
                    pbar.update()

if __name__=="__main__":
    # fp_model = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard/Model_unet/gen_cut128stride100_TruotLo_UINT8_20230515_232218/gen_cut128stride100_TruotLo_UINT8_20230515_232218_luu_cau_truc2.h5'
    fp_model = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard/Model_U2Net_v2/model_128/Model_U2Net_size128_20230515_184214_cau_truc.h5'
    fp_img = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/bo_them_nodata/Img_uint8_crop/S2B_L2A_20190422_Full_0.tif'
    fp_predict = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/bo_them_nodata/Img_uint8_crop/222.tif'
    type_model = ['U2NET','U3NET','UNET']
    size_model = 128
    
    model = tf.keras.models.load_model(fp_model)
    predict(model, fp_img, fp_predict, size_model, type_model[type_model.index['U2NET']])
