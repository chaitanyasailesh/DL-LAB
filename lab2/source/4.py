!pip install tensorboardcolab

from tensorboardcolab import *
tbc=TensorBoardColab()
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

from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('spam1.csv',encoding='latin-1')
# Keeping only the neccessary columns
data = data[['category','message']]

data['message'] = data['message'].apply(lambda x: x.lower())
data['message'] = data['message'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data[data['category'] == 'ham'].size)
print(data[data['category'] == 'spam'].size)

max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['message'].values)
X = tokenizer.texts_to_sequences(data['message'].values)
print(X)
X = pad_sequences(X)
print(X)
embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_features, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
# print(model.summary())

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['category'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = createmodel()

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 5, batch_size=256, verbose = 1,callbacks=[TensorBoardColabCallback(tbc)])


score,accuracy = model.evaluate(X_test,Y_test,verbose=1,batch_size=256)
print(score)
print(accuracy)