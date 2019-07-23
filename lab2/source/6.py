!pip install tensorboardcolab
from tensorboardcolab import *

# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras import backend as K
K.set_image_dim_ordering('th')


tbc=TensorBoardColab()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# normalize inputs from 0-255 to 0.0-1.0
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255.0
x_test = x_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)


# Create the model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# flattening the matrix into vector form
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 6
learningrate = 0.1
decay = learningrate/epochs
sgd = SGD(lr=learningrate, momentum=0.5, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#tensorboard = TensorBoard(log_dir="logs/{}",histogram_freq=0, write_graph=True, write_images=True)


# Fit the model
model.fit(x_train, y_train,validation_data=(x_test, y_test), nb_epoch=epochs, batch_size=64,callbacks=[TensorBoardColabCallback(tbc)])


# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("loss: %.2f%%" % (scores[0]*100))
model.save('./model' + '.h5')