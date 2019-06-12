""""

keras.py

functions to run and train autopilots using keras

"""

from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, Adam, rmsprop

import numpy as np

class KerasPilot:

    def load(self, model_path):
        self.model = load_model(model_path)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=int(steps * (1.0 - train_split) / train_split))
        return hist


class KerasLinear(KerasPilot):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = default_linear()
        else:
            self.model = default_linear()

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasCategorical(KerasPilot):
    """
    Baseline Model
    """
    def __init__(self, model=None, *args, **kwargs):
        super(KerasCategorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_categorical()

            # Load supervised weights
            self.model.load_weights("/home/donkey/sandbox/d2/models/pix_baseline_transfer_all")
            print("Weights load successfully!")

    def run(self, img_arr):

        # Hack to get lane segmentation
        #img_arr = segment_lane(img_arr)

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, __ = self.model.predict(img_arr)
        print("Raw prediction: ", angle_binned)
        #print('throttle', throttle)
        #angle_certainty = max(angle_binned[0])
        #angle_unbinned = dk.utils.linear_unbin(angle_binned)
        print("Raw prediction shape: ", angle_binned[0].shape)
        b = np.argmax(angle_binned[0])
        print("Argmax: ", b)
        angle_unbinned = b * (2 / 14) - 1

        # Constant Throttle for now
        throttle = 0.9

        print("NN output: ", angle_unbinned, throttle)

        return angle_unbinned, throttle

class Keras_Q_Categorical(KerasPilot):
    """
    Use DQN Q Network (zero shot or finetuned)
    """
    def __init__(self, model=None, *args, **kwargs):
        super(Keras_Q_Categorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_Q_categorical()

            # Pretrained Q Network
            self.model.load_weights("/home/donkey/sandbox/d2/models/finetune_q_20190525155106.h5")

            print("Weights load successfully!")

    def run(self, input_state):

        # Hack to get lane segmentation
        #img_arr = segment_lane(img_arr)

        input_state = input_state.reshape(1, input_state.shape[0], input_state.shape[1], input_state.shape[2])
        q_value = self.model.predict(input_state)
        #angle_unbinned = dk.utils.linear_unbin(q_value)
        b = np.argmax(q_value)
        angle_unbinned = b * (2 / 14) - 1

        # Constant Throttle
        throttle = 0.7

        print("NN output: ", angle_unbinned, throttle)

        return angle_unbinned, throttle

class Keras_Simple_Categorical(KerasPilot):
    """
    Use Simple NN Archiecture for faster inference
    """
    def __init__(self, model=None, *args, **kwargs):
        super(Keras_Simple_Categorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = simple_categorical()
            self.model.load_weights("/home/donkey/sandbox/d2/models/single_frame_simple_categorical_20190523225641.h5")
            print("Weights load successfully!")

    def run(self, img_arr):

        # Hack to get lane segmentation
        #img_arr = segment_lane(img_arr)

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, __ = self.model.predict(img_arr)
        print("Raw prediction: ", angle_binned)
        #print('throttle', throttle)
        #angle_certainty = max(angle_binned[0])
        #angle_unbinned = dk.utils.linear_unbin(angle_binned)
        print("Raw prediction shape: ", angle_binned[0].shape)
        b = np.argmax(angle_binned[0])
        print("Argmax: ", b)
        angle_unbinned = b * (2 / 14) - 1

        # Constant Throttle for now
        throttle = 1.0

        print("NN output: ", angle_unbinned, throttle)

        return angle_unbinned, throttle


class Keras_StackedFrame_Categorical(KerasPilot):
    """
    Use GrayScale Stacked Frame as Input
    """
    def __init__(self, model=None, *args, **kwargs):
        super(Keras_StackedFrame_Categorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_categorical(input_dimension=(120,160,4))

            # Load weights
            self.model.load_weights("/home/donkey/sandbox/d2/models/stackedframe_categorical_20190526000230.h5")

            print("Weights load successfully!")

    def run(self, input_state):

        # Hack to get lane segmentation
        #img_arr = segment_lane(img_arr)

        input_state = input_state.reshape(1, input_state.shape[0], input_state.shape[1], input_state.shape[2])
        angle_binned, __ = self.model.predict(input_state)
        print("Raw prediction: ", angle_binned)
        #print('throttle', throttle)
        #angle_certainty = max(angle_binned[0])
        #angle_unbinned = dk.utils.linear_unbin(angle_binned)
        print("Raw prediction shape: ", angle_binned[0].shape)
        b = np.argmax(angle_binned[0])
        print("Argmax: ", b)
        angle_unbinned = b * (2 / 14) - 1

        # Constant Throttle for now
        throttle = 0.7

        print("NN output: ", angle_unbinned, throttle)

        return angle_unbinned, throttle

class Keras_LSTM_Categorical(KerasPilot):
    """
    LSTM with 7 states
    """
    def __init__(self, model=None, *args, **kwargs):
        super(Keras_LSTM_Categorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = lstm_categorical(input_dimension=(7,120,160,3))

            # Load weights
            self.model.load_weights("/home/donkey/sandbox/d2/models/lstm_categorical_20190525154029.h5")

            print("Weights load successfully!")

    def run(self, input_state):

        # Hack to get lane segmentation
        #img_arr = segment_lane(img_arr)

        input_state = input_state.reshape(1, input_state.shape[0], input_state.shape[1], input_state.shape[2], input_state.shape[3])
        angle_binned, __ = self.model.predict(input_state)
        print("Raw prediction: ", angle_binned)
        #print('throttle', throttle)
        #angle_certainty = max(angle_binned[0])
        #angle_unbinned = dk.utils.linear_unbin(angle_binned)
        print("Raw prediction shape: ", angle_binned[0].shape)
        b = np.argmax(angle_binned[0])
        print("Argmax: ", b)
        angle_unbinned = b * (2 / 14) - 1

        # Constant Throttle for now
        throttle = 0.7

        print("NN output: ", angle_unbinned, throttle)

        return angle_unbinned, throttle

def default_categorical(input_dimension=(120,160,3)):

    img_in = Input(shape=(input_dimension), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    #img_in = Input(shape=(120, 160, 1), name='img_in')                         # Take as input grayscale lane segment image
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model


def default_Q_categorical():
    """
    For pretrained DQN Q network
    """

    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(120,160,4)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    # 15 categorical bins for Steering angles
    model.add(Dense(15, activation="linear"))

    # No training
    #adam = Adam(lr=1e-4)
    #model.compile(loss='mse',optimizer=adam)
    print("We finished building the Q model")

    return model

def simple_categorical(input_dimension=(120,160,3)):
    """
    Follow Q network architecture. Faster Inference
    """

    img_in = Input(shape=(input_dimension), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(32, (8,8), strides=(4,4), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (4,4), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(512, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model
    

def lstm_categorical(input_dimension=(7,120,160,3)):

    img_in = Input(shape=(input_dimension), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'))(img_in)
    x = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))(x)
    x = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)

    x = LSTM(512, activation='tanh')(x)

    # Steering Categorical
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)

    # Throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    adam = Adam(lr=0.0001, clipnorm=1.0)
    model.compile(optimizer=adam, loss={'angle_out': 'categorical_crossentropy','throttle_out': 'mean_absolute_error'},loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model


def default_linear():
    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in

    # Convolution2D class name is an alias for Conv2D
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=.1)(x)
    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model
