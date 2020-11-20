'''Modified By: Muahmmad Umer
Created On: 11/19/2020
Performing multiclass text classification using bidirectional LSTM'''



import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import csv


########importing libraries###########
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
print(tf.__version__)
import chardet
import pandas as pd
import numpy as np


#############Reading CSV File and Replacing Stopwords in the file with ''##############
articles = []
labels = []

file ='bbc-text.csv'

############To determine the encoding of the csv file uncomment the following code#########
# with open(file, 'rb') as rawdata:
#     result = chardet.detect(rawdata.read(100000))
# result

print(STOPWORDS)
with open("bbc-text.csv", encoding= 'ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)
print(len(labels))
print(len(articles))

###############Doing train and validation split#########
train_size = int(len(articles) * 0.8)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_articles))
print(len(train_labels))
print(len(validation_articles))
print(len(validation_labels))

###############Tokenizing the training data with the vocabulary size of 10000#######
vocab_size = 10000
sequence_length = 250
oov_tok = '<OOV>'
trunc_type = 'post'
padding_type = 'post'

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok) ####using Tokenizer API
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index #######it will create a dictionary of 10000 words with the text and index pair
dict(list(word_index.items())[0:10]) #####We can see the first 10 common words using this

#######Now convert training data from text to sequences and make all the sequence of equal length using padding and
######sequence_length
train_sequences = tokenizer.texts_to_sequences(train_articles)
print(train_sequences[10]) ######It will show the sequence corresponding to the 11th article
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding=padding_type, truncating=trunc_type)
######Do the same thing to the validation data as well
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=sequence_length, padding=padding_type, truncating=trunc_type)
#######Convert training and validation labels from text to sequence and then covert them to numpy arrays
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

#########Instantiate a bi-directional LSTM model as below#############
embedding_dim = 64
model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 10000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()
##################Compile the model and fit the model on the training data#######################
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10 #########We can increase the number of epochs
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
###########Plotting the validation loss and accuracy#########
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

###################Now do the prediction on the new data###################
txt = ["A WeWork shareholder has taken the company to court over the near-$1.7bn (£1.3bn) leaving package approved for ousted co-founder Adam Neumann. thus a great business."]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=250)
print(padded)
print(padded.size)
pred = model.predict(padded)
labels = ['any','sport', 'bussiness', 'politics', 'tech', 'entertainment']
print(pred, labels[np.argmax(pred)])