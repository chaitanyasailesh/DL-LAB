import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import TensorBoard
import re
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.layers import Flatten
from keras import optimizers


from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam.csv',encoding='latin-1')
# Keeping only the neccessary columns
data = data[['category','message']]

data['message'] = data['message'].apply(lambda x: x.lower())
data['message'] = data['message'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data[data['category'] == 'ham'].size)
print(data[data['category'] == 'spam'].size)

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['message'].values)
X = tokenizer.texts_to_sequences(data['message'].values)


print(X)
X = pad_sequences(X)
print(X)


def createmodel():
    model = Sequential()
    model.add(Embedding(2000, 128, dropout=0.2, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))

    model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(2,activation='sigmoid'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['category'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = createmodel()


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 8, batch_size=40, verbose = 2)

score,acc = model.evaluate(X_test,Y_test,verbose=2,batch_size=40)
print(score)
print(acc)