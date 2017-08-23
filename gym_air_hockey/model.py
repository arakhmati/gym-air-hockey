from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import regularizers
from keras.layers.wrappers import TimeDistributed

def build_model():
    
    l1 = 0.0001
    l2 = 0.00001
  
    model = Sequential([
                ConvLSTM2D(8, 16, activation='relu', name='conv1', input_shape=(None, 128, 128, 3),
                           kernel_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout1'),
                MaxPooling2D(3, name='pool1'),
                Conv2D(16, 16, activation='relu', name='conv2', kernel_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout2'),
                MaxPooling2D(3, name='pool2'),
                Flatten(name='flatten'),
                Dense(256, activation='relu',    name='dense1', kernel_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout3'),
                Dense(128, activation='relu',    name='dense2', kernel_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout4'),
                Dense(64,  activation='relu',    name='dense3', kernel_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout5'),
                Dense(9,   activation='softmax', name='out')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='rmsprop', metrics=['accuracy'])
    return model