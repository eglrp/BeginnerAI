import keras
import lib.keras.yolov2.loss as k_loss
import lib.keras.yolov2.dataset as k_data

def model_to_train(cfg, optimizer=keras.optimizers.SGD(0.0001), anchor_length=5):
    """
    helper function to prepare to train
    para:
        model_path: path to model
        optimizer: keras Optimizer instance. Ex. SGD(0.0001), RMsprop(0.001) etc
        weight_path: where to load weight, default None, use model_path's weight
    return:
        model: use to fit. However input should be [y_pred,y1_, y2_, y3_]
    """
    darknet = keras.models.load_model(cfg["MODEL_PATH"])

    image_size = darknet.layers[0].input_shape[1:3]
    y1_ = keras.layers.Input(shape=(5,))#x,y,w,h
    y2_ = keras.layers.Input(shape=(image_size[0]//32, image_size[1]//32, anchor_length, 1))#object_mask
    y3_ = keras.layers.Input(shape=(image_size[0]//32, image_size[1]//32, anchor_length, 5))#object_value
    image_input = keras.layers.Input(shape=(image_size[0],image_size[1],3))
    y_pred = darknet(image_input)

    loss_out = keras.layers.Lambda(k_loss.loss_function, output_shape=(1,))([y_pred,y1_, y2_, y3_])

    model = keras.models.Model(inputs=[image_input, y1_, y2_, y3_],outputs=[loss_out])

    # TODO: change to other optimizer
    model.compile(optimizer=optimizer,loss=lambda x,y: x*y)

    return model, darknet

def train_model(model_path, cfg):
    """
    Use to train model. use fit_generator by default
    para:
        imageFile: ex:/images/imagelist.txt should have the form. origin in top left
            image_path, x, y, w, h, class
            image_path, x, y, w, h, class
            ...
            ex:
            images/image1, 120, 30, 50, 20, 9
        model_path: path to model
        optimizer: keras Optimizer instance. Ex. SGD(0.0001), RMsprop(0.001) etc
        weight_path: where to load weight, default None, use model_path's weight
        val_imageFile: same as imageFile but is used for validation
    return:
        the original model. that's output is (13, 13, 425)
    """
    model = model_path[0]

    val = None
    try:
        """
        In case of exception, save the model
        """
        model.fit_generator(
            k_data.data_generator(cfg["FILE_PATH"],cfg),
            samples_per_epoch=cfg["SAMPLES_PER_EPOCH"],
            nb_epoch=cfg["EPOCHS"],
            verbose=1,
            validation_data=val,
            nb_val_samples=cfg["NB_VAL_SAMPLES"])
    except Exception as e:
        print(e)
    #model.save_weights('exception.h5')
    model_path[1].save_weights(cfg["SAVE_PATH"])
    return model_path[1]