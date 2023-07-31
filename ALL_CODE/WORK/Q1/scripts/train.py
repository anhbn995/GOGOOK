import os
from osgeo import gdal
import tensorflow as tf

from scripts.eval import monitor_results
from models.metrics import iou, dice_coef
from preprocess.prepare_dataset import data_gen
from preprocess.crop_by_box import main_cut_img
from preprocess.build_mask_ import main_build_mask
from preprocess.crop_data import main_crop_overlap
from models.callback.save_best import SavebestweightsandEarlyStopping

def norm_train(model, optimizer, loss, loss_weights, callbacks , model_metrics, data_path,
               img_path, shp_path, box_path, img_size, num_band, epochs, batch_size, 
               num_class, split_ratios, use_test, use_multi, old_weights, size_crop, stride):
    out_path = data_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    overlap_img = os.path.join(out_path, 'img_cut_crop')
    overlap_mask = os.path.join(out_path, 'mask_cut_crop')
    if os.path.exists(overlap_img) and os.path.exists(overlap_mask):
        pass
    else:
        mask_path = main_build_mask(img_path, shp_path)
        crop_img_path = main_cut_img(img_path, box_path, out_path)
        crop_mask_path = main_cut_img(mask_path, box_path, out_path)
        overlap_img = main_crop_overlap(crop_img_path, size_crop=size_crop, stride=stride)
        overlap_mask = main_crop_overlap(crop_mask_path, size_crop=size_crop, stride=stride)
    
    if use_test:
        train_dataset, valid_dataset, test_dataset, _, _, _ = data_gen(os.path.join(overlap_mask, '*.tif'), img_size=img_size, 
                                                                                batch_size=batch_size, N_CLASSES=num_class, numband=num_band, 
                                                                                split_ratios=split_ratios, test_data=use_test, multi=use_multi)
    else:
        train_dataset, valid_dataset, _, _ = data_gen(os.path.join(overlap_mask, '*.tif'), img_size=img_size, 
                                                            batch_size=batch_size, N_CLASSES=num_class, numband=num_band, 
                                                            split_ratios=split_ratios, test_data=use_test, multi=use_multi)
    
    model.compile(loss=loss, loss_weights=loss_weights,
                  optimizer=optimizer, metrics=model_metrics)

    if old_weights:
        if os.path.exists(old_weights):
            model.load_weights(old_weights)
        else:
            raise Exception("File weight isn't exists, %s"%(old_weights))
    if batch_size >1:
        val_batch_size = int(batch_size/2)
    else:
        val_batch_size = batch_size
        
    history_train = model.fit(train_dataset, batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=callbacks, validation_data=valid_dataset, 
                        validation_batch_size=val_batch_size, use_multiprocessing=True)

    if test_dataset:
        monitor_results(model, test_dataset)
    return history_train

# def custom_train():
#     return

if __name__=="__main__":
    from models.core.att_unet import att_unet
    
    mission = ''
    model_name = 'att_unet'
    img_path = ''
    shp_path = ''
    box_path = ''
    
    epochs = 100
    batch_size = 8
    num_class = 1
    split_ratios = 0.8
    use_test = False
    use_multi = False
    old_weights = None

    filter_num = [64, 128, 256, 512]
    input_size = (128, 128, 4)
    num_band = input_size[-1] 
    img_size = input_size[0]
    size_crop = 128
    stride = 32

    model = att_unet(input_size, filter_num, num_class, stack_num_down=2, stack_num_up=2, activation='ReLU', 
            atten_activation='ReLU', attention='add', output_activation='Softmax', batch_norm=True, pool=True, unpool=True, 
            backbone=None, weights='imagenet', freeze_backbone=False, freeze_batch_norm=False, name=model_name)

    print("Init loss function")
    loss = tf.keras.losses.BinaryCrossentropy()
    losses = [loss]
    loss_weights = []
    for i in len(losses):
        loss_weights.append(1.0)

    print("Init metric function")
    if num_class==1:
        recall = tf.keras.metrics.Recall()
        precision = tf.keras.metrics.Precision()
        model_metrics = [precision, recall, dice_coef, iou, tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
    else:
        recall = tf.keras.metrics.Recall()
        precision = tf.keras.metrics.Precision()
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        model_metrics = [precision, recall, dice_coef, iou, accuracy]

    print("Init optimizer function")
    optimizer = tf.keras.optimizers.Adam()

    print("Init callback function")
    def lr_decay(epoch):
        initial_learningrate=1e-3
        if epoch < 1:
            return initial_learningrate
        else:
            return initial_learningrate * 0.9 ** (epoch)
    
    data_path = os.path.join('/home/quyet/DATA_ML/Projects/segmentation/data', mission)
    checkpoint_filepath= '/home/quyet/DATA_ML/Projects/segmentation/logs/tmp'
    log_dir = '/home/quyet/DATA_ML/Projects/segmentation/logs/graph'
    weights_path = '/home/quyet/DATA_ML/Projects/segmentation/weights/%s/'%(model_name)+model_name+'_'+mission+'_'+str(img_size)+'_'+str(num_class)+'class.h5'
    patience = 10
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only= True, 
                                                                    monitor='val_loss', mode='min', save_best_only=True)
    model_lrscheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=1)
    model_lrreduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, min_lr=1e-7, verbose=1)
    model_earlystopping_callback = SavebestweightsandEarlyStopping(patience=patience, weights_path=weights_path)
    model_endtrainnan_callback = tf.keras.callbacks.TerminateOnNaN()
    model_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    model_callbacks = [model_checkpoint_callback, model_lrscheduler_callback,
                        model_lrreduce_callback, model_earlystopping_callback,
                        model_tensorboard_callback,]

    history = norm_train(model, optimizer, losses, loss_weights, model_callbacks , model_metrics, data_path, 
                        img_path, shp_path, box_path, img_size, num_band, epochs, batch_size, 
                        num_class, split_ratios, use_test, use_multi, old_weights, size_crop, stride)