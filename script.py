import pandas as pd
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, Dropout
import matplotlib.pyplot as plt

#%% Import Data
train_raw = pd.read_csv('train.csv')
train_raw.head()
train = train_raw.copy()
train.text = train.text.replace('@\w+', '', regex=True)
train.text = train.text.replace('''[\\./[\]$%*,'"?\-+=!`~|^&()><;:]''', '', regex=True)
train.text = train.text.replace('%20', ' ', regex=True)
train.text = train.text.str.lower()
train.keyword = train.keyword.replace('%20', ' ', regex=True)
train['hashtag'] = train.text.str.lower().str.extract('(#\w+)', expand=False)

#%% Prepare Tokenizer
tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(train.text.values)
sequence = tokenizer.texts_to_sequences(train.text.values)
one_hot_result = tokenizer.texts_to_matrix(train.text.values, mode='binary')
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# Cut off after 100 words.
data = pad_sequences(sequence, maxlen=100)
labels = np.asarray(train.target)
print ('Shape of data tensor:', data.shape)
print ('Shape of labels tensor:', labels.shape)
# Split the data into train and test sets.
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
train_samples = round(0.3*len(data))
validation_samples = len(data) - train_samples
x_train = data[:train_samples]
y_train = labels[:train_samples]
x_val = data[train_samples: train_samples + validation_samples]
y_val = labels[train_samples: train_samples + validation_samples]

#%% Prepare gloVe word-embeddings file.
glove_dir = 'glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),  encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print ('Found %s word vectors.' % len(embeddings_index))

#%% Prepare gloVe matrix
max_words = 30000
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

#%% RNN Model
model = Sequential()
model.add(Embedding(max_words, embedding_dim))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(data, labels,  epochs=4, batch_size=16, validation_split=0.4)

#%% Plotting Result
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#%% Evaluate
# Toekenizing test set
test_raw = pd.read_csv('test.csv')
sequences = tokenizer.texts_to_sequences(test_raw.text)
x_test = pad_sequences(sequences)
target = model.predict(x_test)
target = np.round(target).astype(int)
result = test_raw.id.to_frame()
result['target'] = target
result.to_csv('20200212_submission.csv', index=None)
