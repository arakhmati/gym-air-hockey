from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import regularizers

def build_model():
    
    l1 = 0.0000
    l2 = 0.01
  
    model = Sequential([
                Conv2D(16, 16, activation='relu', name='conv1', input_shape=(128, 128, 3), 
                       kernel_regularizer=regularizers.l2(l2), activity_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout1'),
                MaxPooling2D(3, name='pool1'),
                Conv2D(32, 16, activation='relu', name='conv2', 
                       kernel_regularizer=regularizers.l2(l2), activity_regularizer=regularizers.l1(l1)),
#                Dropout(0.1, name='dropout2'),
                MaxPooling2D(3, name='pool2'),
                Flatten(name='flatten'),
                Dense(256, activation='relu',    name='dense1', 
                       kernel_regularizer=regularizers.l2(l2), activity_regularizer=regularizers.l1(l1)),
                Dropout(0.2, name='dropout4'),
                Dense(128, activation='relu',    name='dense2', 
                       kernel_regularizer=regularizers.l2(l2), activity_regularizer=regularizers.l1(l1)),
#                Dropout(0.1, name='dropout5'),
                Dense(64,  activation='relu',    name='dense3', 
                       kernel_regularizer=regularizers.l2(l2), activity_regularizer=regularizers.l1(l1)),
#                Dropout(0.1, name='dropout6'),
                Dense(9,   activation='softmax', name='out')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model