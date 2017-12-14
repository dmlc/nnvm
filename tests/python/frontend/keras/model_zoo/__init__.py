import keras

def get_vgg16():
    return keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
        input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)

def get_xception():
    return keras.applications.xception.Xception(include_top=True, weights='imagenet',
        input_tensor=None, input_shape=(299,299,3), pooling=None, classes=1000)

def get_mobilenet():
    return keras.applications.mobilenet.MobileNet(input_shape=(224,224,3), alpha=1.0,
        depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet',
        input_tensor=None, pooling=None, classes=1000)

def get_resnet50():
    return keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet',
        input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)

# for debug purpose
def get_mlp():
    from keras.models import Model
    from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
    img_input = Input(shape=(224,224,3))
    x = MaxPooling2D((2,2), strides=(2,2))(img_input)
    x = AveragePooling2D((2,2), strides=(2,2))(x)
    x = Conv2D(9, (3,3), activation='tanh', strides=(1, 1), padding='same', kernel_initializer='uniform')(x)
    x = BatchNormalization(beta_initializer='uniform', gamma_initializer='uniform', moving_mean_initializer='uniform', moving_variance_initializer='ones')(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu', kernel_initializer='uniform')(x)
    model = Model(img_input, x)
    return model
