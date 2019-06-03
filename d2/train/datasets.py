"""
Prepare data for model training
"""

import os
import glob
import cv2
import json
import numpy as np
import pandas as pd
from numpy.random import permutation
from sklearn.model_selection import train_test_split

rand_seed = 18

class Datasets(object):

    @staticmethod
    def make_single_frame_data(folder_path, num_frames=4):
        """
        make single frame data to train model
        """
        train_data = []  
        steering_data = []
        throttle_data = []
        for idx, img_path in enumerate(Datasets.img_generator(folder_path)):
            label_path = os.path.join(folder_path, 'record_' + img_path.split('_')[0] + '.json')
            img_path = os.path.join(folder_path, img_path)
            print('processing img: ', img_path)
            print('label path: ', label_path)

            # read image
            img = cv2.imread(img_path)
            img = Datasets.process_image(img, grayscale=False)

            # read label
            label_json = Datasets.process_label(label_path)
            steering = label_json['user/angle']
            steering_bin = linear_bin(steering)
            throttle = label_json['user/throttle']
            #print("steering angle: ", steering)
            #print("steering bin: ", steering_bin)
            #print("throttle: ", throttle)
            steering_data.append(steering_bin)
            throttle_data.append(throttle)
            train_data.append(img)

        X_train, X_valid, Y_train, Y_valid = Datasets.shuffle_and_split_data(train_data, steering_data, throttle_data, 0.1)

        return X_train, X_valid, Y_train, Y_valid

    @staticmethod
    def make_stacked_frame_data(folder_path, num_frames=4):
        """
        create grayscale stacked frame to train model
        """
        train_data = []  
        steering_data = []
        throttle_data = []
        for idx, img_path in enumerate(Datasets.img_generator(folder_path)):
            label_path = os.path.join(folder_path, 'record_' + img_path.split('_')[0] + '.json')
            img_path = os.path.join(folder_path, img_path)
            print('processing img: ', img_path)
            print('label path: ', label_path)

            # read image
            img = cv2.imread(img_path)
            img = Datasets.process_image(img)

            # read label
            label_json = Datasets.process_label(label_path)
            steering = label_json['user/angle']
            steering_bin = linear_bin(steering)
            throttle = label_json['user/throttle']
            #print("steering angle: ", steering)
            #print("steering bin: ", steering_bin)
            #print("throttle: ", throttle)
            steering_data.append(steering_bin)
            throttle_data.append(throttle)
            if idx == 0:
                s_img = np.stack(([img]*num_frames),axis=2)
                #s_img = s_img.reshape(1, s_img.shape[0], s_img.shape[1], s_img.shape[2])
            else:
                img = img.reshape(img.shape[0], img.shape[1], 1)
                s_img = np.append(img, s_img[:, :, :(num_frames-1)], axis=2)
            train_data.append(s_img)

        X_train, X_valid, Y_train, Y_valid = Datasets.shuffle_and_split_data(train_data, steering_data, throttle_data, 0.1)

        return X_train, X_valid, Y_train, Y_valid

    @staticmethod
    def make_lstm_data(folder_path, num_states=7):
        """
        create training data for LSTM model
        """
        train_data = []  
        steering_data = []
        throttle_data = []
        for idx, img_path in enumerate(Datasets.img_generator(folder_path)):
            label_path = os.path.join(folder_path, 'record_' + img_path.split('_')[0] + '.json')
            img_path = os.path.join(folder_path, img_path)
            print('processing img: ', img_path)
            print('label path: ', label_path)

            # read image
            img = cv2.imread(img_path)
            img = Datasets.process_image(img, grayscale=False)

            # read label
            label_json = Datasets.process_label(label_path)
            steering = label_json['user/angle']
            steering_bin = linear_bin(steering)
            throttle = label_json['user/throttle']
            #print("steering angle: ", steering)
            #print("steering bin: ", steering_bin)
            #print("throttle: ", throttle)
            steering_data.append(steering_bin)
            throttle_data.append(throttle)
            if idx == 0:
                s_img = np.stack(([img]*num_states), axis=0)
                #s_img = s_img.expand_dims(s_img, axis=0)
            else:
                img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
                s_img = np.append(img, s_img[:(num_states-1), :, :, :], axis=0)
            train_data.append(s_img)

        X_train, X_valid, Y_train, Y_valid = Datasets.shuffle_and_split_data(train_data, steering_data, throttle_data, 0.1)

        return X_train, X_valid, Y_train, Y_valid

    def shuffle_and_split_data(train_data, steering_data, throttle_data, validation_size=0.1):

        num_samples = len(train_data)
        train_data = np.array(train_data)
        steering_data = np.array(steering_data)
        throttle_data = np.array(throttle_data)

        perm = permutation(num_samples)
        train_data = train_data[perm]
        steering_data = steering_data[perm]
        throttle_data = throttle_data[perm]

        validation_size = 0.1
        num_training_samples = int(num_samples * (1-validation_size))
        X_train = train_data[:num_training_samples]
        X_valid = train_data[num_training_samples:]
        Y_steering_train = steering_data[:num_training_samples]
        Y_steering_valid = steering_data[num_training_samples:]
        Y_throttle_train = throttle_data[:num_training_samples]
        Y_throttle_valid = throttle_data[num_training_samples:]

        return X_train, X_valid, [Y_steering_train, Y_throttle_train], [Y_steering_valid, Y_throttle_valid]


    def img_generator(folder_path):
        img_path = os.path.join(folder_path, '*.jpg')
        for i in range(len(glob.glob(img_path))):
            yield str(i) + "_cam-image_array_.jpg"

    def process_image(img, dimension=(160,120), resize=True, grayscale=True, threshold=True):
        if resize:
            img =  cv2.resize(img, dimension)
        if threshold:
            hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            sensitivity = 100
            lower_white = np.array([0,0,255-sensitivity])
            upper_white = np.array([255,sensitivity,255])
            white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

            mean_pixel = np.mean(img, axis=(0, 1))
            #print("Mean Pixel:", mean_pixel)

            white_mask_inv = cv2.bitwise_not(white_mask)
            no_light_image = cv2.bitwise_and(img, img, mask=white_mask_inv)
            r_layer = np.full((img.shape[0], img.shape[1], 1), np.uint8(mean_pixel[0]))
            g_layer = np.full((img.shape[0], img.shape[1], 1), np.uint8(mean_pixel[1]))
            b_layer = np.full((img.shape[0], img.shape[1], 1), np.uint8(mean_pixel[2]))
            mode_image = np.concatenate((r_layer, g_layer, b_layer), axis=2)
            filled_image = cv2.bitwise_and(mode_image, mode_image, mask=white_mask)
            filtered_image = cv2.add(no_light_image, filled_image)
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return img

    def process_label(label_path):
        with open(label_path, 'r') as f:
            return json.load(f)

def linear_bin(a):
    """
    Convert a value to a categorical array.

    Parameters
    ----------
    a : int or float
        A value between -1 and 1

    Returns
    -------
    list of int
        A list of length 15 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 14))
    arr = np.zeros(15)
    arr[int(b)] = 1
    return arr
    


if __name__ == '__main__':
    folder_path = '../data/tub_5_19-05-21'
    Datasets.make_stacked_frame_data(folder_path)
    #Datasets.make_lstm_data(folder_path)
    #Datasets.make_single_frame_data(folder_path)

