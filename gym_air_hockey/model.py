from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import regularizers

def conv_model(l1=0.00000, l2=0.00000):
    model = Sequential([
                Conv2D(8, 12, activation='relu', name='conv1', input_shape=(128, 128, 3)),
                MaxPooling2D(2, name='pool1'),
                Conv2D(12, 7, activation='relu', name='conv2'),
                MaxPooling2D(2, name='pool2'),
                Conv2D(16, 5, activation='relu', name='conv3'),
                MaxPooling2D(2, name='pool4'),
                Flatten(name='flatten'),
                Dense(128, activation='relu',    name='dense1'),
                Dense(512, activation='relu',    name='dense2'),
                Dense(256, activation='relu',    name='dense3'),
                Dense(128,  activation='relu',    name='dense4'),
                Dense(9,   activation='softmax', name='out')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model

def convlstm_model(l1=0.00000, l2=0.00000):
    model = Sequential([
                ConvLSTM2D(8, 12, activation='relu', name='conv1', input_shape=(None, 128, 128, 3)),
                MaxPooling2D(2, name='pool1'),
                Conv2D(12, 7, activation='relu', name='conv2'),
                MaxPooling2D(2, name='pool2'),
                Conv2D(16, 5, activation='relu', name='conv3'),
                MaxPooling2D(2, name='pool4'),
                Flatten(name='flatten'),
                Dense(512, activation='relu',    name='dense1'),
                Dense(256, activation='relu',    name='dense2'),
                Dense(128, activation='relu',    name='dense3'),
                Dense(64,  activation='relu',    name='dense4'),
                Dense(9,   activation='softmax', name='out')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model