# Code used for Pixmoving Hackathon 2019

## Background

Code used by Felix during [Pixmoving Hackathon](https://www.pixmoving.com/movinghackathon). This repo just contains the code but not the dataset nor the trained models.

## Description

Here is a high level description of the important files:

`d2/manage.py` - Default manage.py file

`d2/manage_enhanced.py` - Include support for Stacked Frame, and Time Sequence Frames (for LSTM) imported as parts 

`d2/manage_enhanced_cv.py` - Include Image Thresholding to eliminate light glare, also implemented as parts

`d2/train/datasets.py` - Dataloader to prepare training data for default model, stacked frame model, and LSTM model. 

`d2/train/train.py` - Train various models such as LSTM, fine-tune Q model from RL, default CNN with grayscale stacked frame, etc

`donkeycar/donkeycar/parts/cv.py` - CV related algorithms implemented as parts. Image Thresholding, Stacked Frame, and Time Sequence Frames code can be found here

`donkeycar/donkeycar/parts/keras.py` - Wrap Keras models (such as LSTM, Q model, default CNN) as parts

## Dependencies

Run on Jetson Nano following [this guide](https://medium.com/@feicheung2016/getting-started-with-jetson-nano-and-autonomous-donkey-car-d4f25bbd1c83). 
