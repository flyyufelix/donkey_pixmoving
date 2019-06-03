import datetime

from keras.layers import Input, Dense, merge
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Adam, rmsprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.activations import softmax
from keras.models import Sequential

from datasets import Datasets

data_folder_path = '../data/tub_05_26_3'

def train_single_frame_model():
    """
    Same as default categorical from donkey
    """

    X_train, X_valid, Y_train, Y_valid = Datasets.make_single_frame_data(data_folder_path)

    model = default_categorical()

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
                 #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_data=(X_valid, Y_valid))

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save_weights('saved_models/single_frame_categorical_' + timestamp + '.h5', overwrite=True)

def train_single_frame_simple_model():

    X_train, X_valid, Y_train, Y_valid = Datasets.make_single_frame_data(data_folder_path)

    model = simple_categorical()

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
                 #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_data=(X_valid, Y_valid))

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save_weights('saved_models/single_frame_simple_categorical_' + timestamp + '.h5', overwrite=True)

def train_stacked_frame_model():

    X_train, X_valid, Y_train, Y_valid = Datasets.make_stacked_frame_data(data_folder_path)

    model = default_categorical(input_dimension=(120,160,4))

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
                 #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_valid, Y_valid))

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save_weights('saved_models/stackedframe_categorical_' + timestamp + '.h5', overwrite=True)

def train_stacked_frame_simple_model():

    X_train, X_valid, Y_train, Y_valid = Datasets.make_stacked_frame_data(data_folder_path)

    model = simple_categorical(input_dimension=(120,160,4))

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
                 #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train, Y_train, epochs=3, batch_size=64, validation_data=(X_valid, Y_valid))

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save_weights('saved_models/stackedframe_simple_categorical_' + timestamp + '.h5', overwrite=True)

def train_lstm_model():

    X_train, X_valid, Y_train, Y_valid = Datasets.make_lstm_data(data_folder_path)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
                 #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model = lstm_categorical(input_dimension=(7,120,160,3))
    model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_valid, Y_valid))

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save_weights('saved_models/lstm_categorical_' + timestamp + '.h5', overwrite=True)

def finetune_q_model():

    X_train, X_valid, Y_train, Y_valid = Datasets.make_stacked_frame_data(data_folder_path)

    # Only need steering targets
    Y_train = Y_train[0] 
    Y_valid = Y_valid[0]

    model = q_categorical(input_dimension=(120,160,4))

    # Load pretrained Q model for finetuning
    model.load_weights('saved_models/robin_track_v2_highres.h5')
    model.layers[-1].activation = softmax
    adam = Adam(lr=1e-4) # Use a smaller learning rate for fine-tuning?
    model.compile(loss='categorical_crossentropy',optimizer=adam)
    print("weights Load Successfully!")

    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
                 #ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    model.fit(X_train, Y_train, epochs=20, batch_size=64, validation_data=(X_valid, Y_valid))

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model.save_weights('saved_models/finetune_q_' + timestamp + '.h5', overwrite=True)

def simple_categorical(input_dimension=(120,160,3)):
    """
    Follow Q network architecture
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

def q_categorical(input_dimension=(120,160,4)):
    """
    DQN Q network for finetuning
    """
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(input_dimension)))  
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

    return model

def default_categorical(input_dimension=(120,160,3)):

    img_in = Input(shape=(input_dimension), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
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

if __name__ == '__main__':
    #train_single_frame_model()
    #train_single_frame_simple_model()
    train_stacked_frame_model()
    #train_stacked_frame_simple_model()
    #train_lstm_model()
    #finetune_q_model()
