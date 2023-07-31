import tensorflow as tf
from models.import_module import Model_U2Net

my_model = Model_U2Net(512,3)
# my_model.load_weights('/home/skm/SKM16/Work/OpenLand/U2net/mm//unet_openland.h5')
# my_model.save('/home/skm/SKM16/Work/OpenLand/U2net/mm/unet_openland_loaded.h5')


# my_model.load_weights('/home/skm/SKM16/ALL_MODEL/Openland/logslogs_Roads_of_Openland_1666240261/weight/Road_of_Openland_1666240261.h5')
# my_model.save('/home/skm/SKM16/ALL_MODEL/Openland/logslogs_Roads_of_Openland_1666240261/weight/Road_of_Openland_1666240261_loadmodel.h5')

my_model.load_weights(r'/home/skm/SKM16/ALL_MODEL/Openland/logs_Water_of_Openland_V2_1666753511/weight/Water_of_Openland_V2_1666753512.h5')
my_model.save(r'/home/skm/SKM16/ALL_MODEL/Openland/logs_Water_of_Openland_V2_1666753511/weight/Water_of_Openland_V2_1666753512_loadmodel.h5')