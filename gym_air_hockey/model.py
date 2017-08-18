from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def build_model():
  
    model = Sequential([
                Conv2D(8, 16, activation='relu', name='conv1', input_shape=(128, 128, 3)),
                MaxPooling2D(2, name='pool1'),
                Conv2D(16, 16, activation='relu', name='conv2'),
                MaxPooling2D(2, name='pool2'),
                Conv2D(5, 5, activation='relu', name='conv3'),
                MaxPooling2D(2, name='pool3'),
                Flatten(name='flatten'),
                Dense(256, activation='relu',    name='dense1'),
                Dense(128, activation='relu',    name='dense2'),
                Dense(32,  activation='relu',    name='dense3'),
                Dense(9,   activation='softmax', name='softmax')
                ])
    model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])
    return model


"""
input 128 128
conv  123 123
pool   61  61
conv   46  46
pool   23  23
conv   19  19
pool    8   8
"""