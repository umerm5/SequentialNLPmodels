'''Author: Muahmmad Umer
Created On: 11/19/2020
Performing tag predictions from programming questions on stack overflow using Embedding'''



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
import tarfile


###############Reading and untaring the tar file, extracting into a directory as below###############
import tarfile
dataset = tarfile.open('stack_overflow_16k.tar.gz')
dataset.extractall('/SO')
os.listdir('/SO')
train_dir = os.path.join('/SO', 'train')
os.listdir(train_dir)
#########Read a sample file in the train directory as below#########
sample_file = os.path.join(train_dir, 'csharp/1999.txt')
with open(sample_file) as f:
  print(f.read())

###########Use "text_dataset_from_directory" from keras to make training and validation batches from the training data########
batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    '/SO/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    '/SO/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

###########Also read the test data as follows#########
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    '/SO/test',
    batch_size=batch_size)

##########Now use a text vectorize layer to convert text to sequences, first we instantiate that layer and then adapt this layer
#########to the training data

max_features = 10000 ###########vocabulary size
sequence_length = 250 ########maximum sequence length

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 codes and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_code, first_label = text_batch[0], label_batch[0]
print("Code", first_code)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized Code", vectorize_text(first_code, first_label))

# print("1 ---> ",vectorize_layer.get_vocabulary()[1])
# print(" 36 ---> ",vectorize_layer.get_vocabulary()[36])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

###########Now map all training, validation, and test raw text data to sequences in order to pass them to the LSTM
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#########Following is done for proper memory use############
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


##########We have the data ready to pass to the embedding layer###########

#########Instantiate the model################
embedding_dim = 32
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim, name= 'embedding'),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(4)])

model.summary()

model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


################Now export model for production and deployment purposes########
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

