import tensorflow as tf
from models.import_module import Model_U2Net,Model_UNet3plus

my_model = Model_U2Net(512,3)


my_model.load_weights(r'/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/Pond_512_fix/logsfix/u2net_512_Pond_V1_fix.h5')
my_model.save(r'/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/Pond_512_fix/logsfix/u2net_512_Pond_V1_fix_model_ok.h5')
