"""
https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
https://github.com/guptajay/Kaggle-Digit-Recognizer

Overfitting is fine, only needs to recognize the same 10 digits, format of the digits will not change
"""
from os import path, cpu_count, system, name

from PIL import Image
from numpy import array, concatenate
from loguru import logger
import tensorflow as tf
print("TENSORFLOW VERSION", tf.__version__)

from tensorflow import keras
from tensorflow.config.threading import set_intra_op_parallelism_threads as cpu_config
from tensorflow.config.experimental import list_physical_devices, set_memory_growth as gpu_config
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from util.util import get_labels, standardize_images
from logs.log import setup_logger

class Active:
    def __init__(self):
        self.CPU=0
        self.GPU=1
        self.CUSTOM_SET=1
        self.KERAS_SET=1
        self.train_images=None
        self.train_labels=None
        self.test_images=None
        self.test_labels=None

    def set_cpu(self):
        threads = cpu_count() 
        logger.info(f"Using {threads} threads")
        if threads == 0:
            logger.error("invalid cpu count")
            self.CPU = 0
            return False
        cpu_config(threads)
        cpu_config(threads)
        self.CPU = 1
        logger.info("Using CPU")
        return True

    def set_gpu(self):
        logger.info("Using GPU")
        gpus = list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    gpu_config(gpu, True) 
            except RuntimeError as e:
                self.GPU = 0
                logger.error(e)
                return False
        else:
            self.GPU = 0
            logger.error("invalid gpu count")
            return False
        self.GPU = 1
        logger.info("Using GPU")
        return True

    def set_hardware(self):
        if self.CPU == 0 and self.GPU == 1:
            if not self.set_gpu():
                logger.error("invalid gpu configuration")
                logger.debug("trying cpu ...")
                if not self.set_cpu():
                    logger.error("invalid cpu and gpu configuration")
                    raise Exception("Invalid hardware configuration")
        elif self.CPU == 1 and self.GPU == 0:
            if not self.set_cpu():
                logger.error("invalid cpu configuration")
                logger.debug("trying gpu ...")
                if not self.set_gpu():
                    logger.error("invalid cpu and gpu configuration")
                    raise Exception("Invalid hardware configuration")
        else:
            logger.error("invalid hardware configuration")
            raise Exception("Invalid hardware configuration")


    def custom_set(self, extra_set='./extra_set', width=28, height=28, batch_size=32):
        custom_images_list, custom_labels_list, augmented_images_list, augmented_labels_list = [], [], [], []
        custom_labels_dict = get_labels(extra_set)
        datagen = ImageDataGenerator(
        rotation_range=10,      
        zoom_range=0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1 
        )

        for img_name, label in custom_labels_dict.items():
            img_path = path.join(extra_set, img_name)
            if path.exists(img_path):  
                img = Image.open(img_path).convert('L')  
                img = img.resize(width, height)  
                img_array = array(img)
                custom_images_list.append(img_array)
                custom_labels_list.append(int(label))
            else:
                logger.error(f"invalid image path: {img_path}")
                raise Exception("Invalid image path")  
        logger.info(f"loaded images in: {extra_set}")

        custom_images = array(custom_images_list).astype('float32')
        custom_images = standardize_images(custom_images)
        custom_images = custom_images.reshape((len(custom_images_list), width, height, 1))

        for x_batch, y_batch in datagen.flow(custom_images, custom_labels_list, batch_size=batch_size):
            augmented_images_list.extend(x_batch)
            augmented_labels_list.extend(y_batch)
            if len(augmented_images_list) >= 5 * len(custom_images_list):
                break

        augmented_images = array(augmented_images_list).astype('float32')
        augmented_images = standardize_images(augmented_images)
        augmented_labels_encoded = keras.utils.to_categorical(augmented_labels_list, num_classes=10)

        if self.KERAS_SET == 1 and self.CUSTOM_SET == 1:
            self.keras_set()
            self.train_images = concatenate([self.train_images, augmented_images], axis=0)
            self.train_labels = concatenate([self.train_labels, augmented_labels_encoded], axis=0)
            self.test_images = concatenate([self.test_images, augmented_images], axis=0)
            self.test_labels = concatenate([self.test_labels, augmented_labels_encoded], axis=0)
            logger.info("custom set concatenated with keras set")
        elif self.KERAS_SET == 0 and self.CUSTOM_SET == 1:
            self.train_images = augmented_images
            self.train_labels = augmented_labels_encoded
            self.test_images = augmented_images
            self.test_labels = augmented_labels_encoded
            logger.info("custom set loaded")
        else:
            logger.error("invalid set configuration, uncaught error")
            raise Exception("Invalid set configuration")
    
    def keras_set(self, width=28, height=28):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        train_images = standardize_images(train_images)
        test_images = standardize_images(test_images)

        self.train_images = train_images.reshape((60000, width, height, 1))
        self.test_images = test_images.reshape((10000, width, height, 1))
        self.train_labels = keras.utils.to_categorical(train_labels)
        self.test_labels = keras.utils.to_categorical(test_labels)

        logger.info("keras set loaded")

    def load_data(self):
        if self.CUSTOM_SET == 1:
            self.custom_set()
        if self.KERAS_SET == 1:
            self.keras_set()

    def train(self, 
              lr=None,
              base=32,
              kernel_size=5,
              strides=1,
              activation='relu',
              input_shape=(28, 28, 1),
              kr=l2(0.0005),
              use_bias=False,
              pool_size=2,
              dropout=0.25,
              _default=10,
              _activation='softmax',
              epochs=30, 
              batch_size=64,
              checkpoint = ModelCheckpoint('./trained/modelx.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
              variable_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
            ):
        

        system('clear' if name == 'posix' else 'cls')
        logger.info("training model")

        model = Sequential([
            Conv2D(base, kernel_size=kernel_size, strides=strides, activation=activation, input_shape=input_shape, kernel_regularizer=kr),
            Conv2D(base, kernel_size=kernel_size, strides=strides, use_bias=use_bias),
            BatchNormalization(),
            Activation(activation),
            MaxPooling2D(pool_size=pool_size, strides=strides*2),
            Dropout(dropout),
            Conv2D(base*2, kernel_size=kernel_size-2, strides=strides, activation=activation, kernel_regularizer=kr),
            Conv2D(base*2, kernel_size=kernel_size-2, strides=strides, use_bias=use_bias),
            BatchNormalization(),
            Activation(activation),
            MaxPooling2D(pool_size=pool_size, strides=strides*2),
            Dropout(dropout),
            Flatten(),
            Dense(base*8, use_bias=use_bias),
            BatchNormalization(),
            Activation(activation),
            Dense(base*4, use_bias=use_bias),
            BatchNormalization(),
            Activation(activation),
            Dense(base*2, use_bias=use_bias),
            BatchNormalization(),
            Activation(activation),
            Dropout(dropout),
            Dense(_default, activation=_activation)
        ])
        
        model.compile(optimizer=Adam(learning_rate=lr) if lr else 'adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        model.fit(self.train_images, self.train_labels, 
                  epochs=epochs, 
                  batch_size=batch_size, 
                  validation_data=(self.test_images, self.test_labels), 
                  callbacks=[checkpoint, variable_learning_rate])

"""------------------------------------------------------------------------------------------------------------------------------------------------"""

setup_logger()
model = Active()
model.set_hardware()
model.load_data()
model.train()