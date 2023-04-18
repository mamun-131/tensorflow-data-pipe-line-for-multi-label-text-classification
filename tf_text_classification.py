# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:26:47 2023

@author: Md Mamunur Rahman
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import pickle
import os.path
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np


############ Data Pipeline Starts #############################
#instentiate tokenizer
tokenizer = tfds.deprecated.text.Tokenizer()

#import data from files
data1 = tf.data.TextLineDataset("data1.csv").skip(1)
data2 = tf.data.TextLineDataset("data2.csv").skip(1)
data3 = tf.data.TextLineDataset("data3.csv").skip(1)
dataset = data1.concatenate(data2).concatenate(data3)

#show data example
for line in dataset.take(5):
    text_data=tf.strings.split(line,",", maxsplit=3)[3]
    #print(text_data.numpy())

#TODO list for data pipeline
#1. Create Vocabulary
#2. Token text encoding
#3. Padding the batch

def filter_train_data(line):
    splited_data = tf.strings.split(line,",",maxsplit=3)
    data_type = splited_data[1] #train, test
    label_data = splited_data[2] # love, life, god
    return(
        True
        if data_type == 'train'
        else
        False
        )
def filter_test_data(line):
    splited_data = tf.strings.split(line,",",maxsplit=3)
    data_type = splited_data[1] #train, test
    label_data = splited_data[2] # love, life, god
    return(
        True
        if data_type == 'test'
        else
        False
        )
def filter_pred_data(line):
    splited_data = tf.strings.split(line,",",maxsplit=3)
    data_type = splited_data[1] #train, test
    label_data = splited_data[2] # love, life, god
    return(
        True
        if data_type == 'pred'
        else
        False
        )

ds_train = dataset.filter(filter_train_data)    
ds_test = dataset.filter(filter_test_data)    
ds_pred = dataset.filter(filter_pred_data) 
#print(ds_pred)

#show data example
pred_data = []
for line in ds_pred:
    #print(line)
    text_data=tf.strings.split(line,",", maxsplit=3)[3]
    pred_data.append(text_data.numpy())
    #print(text_data.numpy())

#building vocabulary
def create_vocabulary(ds_train, threshold=3):
    word_frequency = {}
    vocabulary = set()
    vocabulary.update(["sostoken"])
    vocabulary.update(["eostoken"])    
    
    #Tokenizing
    for line in ds_train:
        split_data = tf.strings.split(line,",",maxsplit=3)
        text = split_data[3]
        tokenized_data = tokenizer.tokenize(text.numpy().lower())
        #print(tokenized_data)
        for word in tokenized_data:
            if word not in word_frequency:
                word_frequency[word] = 1
            else:
                word_frequency[word] +=1
            # print(word_frequency[word])
            # print(word)
            #check threshold
            if word_frequency[word] == threshold:
                vocabulary.update(tokenized_data)
    #print(vocabulary)
    return vocabulary

#save or load vocabular
def load_save_vocabulary(dstrain, rewrite = False):
    path = './vocab_file.obj'
    check_file = os.path.exists(path)
    if check_file and rewrite == False:
        vocab_file = open("vocab_file.obj","rb")
        vocabulary = pickle.load(vocab_file)
    else:
        vocabulary = create_vocabulary(dstrain)
        vocab_file = open("vocab_file.obj","wb")
        pickle.dump(vocabulary,vocab_file)
    
    return vocabulary
        
vocabulary = load_save_vocabulary(ds_train)            

#print(vocabulary)    
    
#Numericalaize or building encoder 
encoder = tfds.deprecated.text.TokenTextEncoder(
    list(vocabulary), oov_token = "<UNK>", lowercase = True, tokenizer = tokenizer
    )

def encoder_action(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_mapping(line):
    split_line = tf.strings.split(line,",",maxsplit=3)
    text = "sostoken " + split_line[3] + " eostoken"
    label_text = split_line[2]
    if label_text == "love":
        label = 0
    elif label_text == "life":
        label = 1
    elif label_text == "god":
        label = 2
    else:
        label = 3
        
    (encoded_text, label) = tf.py_function(
        encoder_action, inp=[text, label], Tout=(tf.int64, tf.int32),
        )
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(encode_mapping,num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.shuffle(100)
ds_train = ds_train.padded_batch(16, padded_shapes = ([None], ()))

ds_test = ds_test.map(encode_mapping)
ds_test = ds_test.padded_batch(16, padded_shapes = ([None], ()))

############ Data Pipeline Ends #############################
    
############ Model Development and training Starts ##########
#TODO
# model defination
# model.complie
# model.fit
# model.evaluate 

model = keras. Sequential(
    [
     layers.Masking(mask_value=0),
     layers.Embedding(input_dim=len(vocabulary)+2,output_dim=32),
     layers.GlobalAveragePooling1D(),
     layers.Dense(64, activation = "relu"),
     layers.Dense(32, activation = "relu"),
     layers.Dense(4, activation='softmax')
     ]
    )

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                optimizer=keras.optimizers.Adam(3e-4, clipnorm=1), 
                metrics=["accuracy"] )

model.fit(ds_train, epochs=100)
model.evaluate(ds_test)
model.save(os.path.join('models','quote_classification.h5'))

loaded_model = load_model(os.path.join('models/', 'quote_classification.h5'))


## result unclassified example for prediction

ds_pred = ds_pred.map(encode_mapping)
ds_pred = ds_pred.padded_batch(16, padded_shapes = ([None], ()))

predictions = loaded_model.predict(ds_pred, verbose=2)
aa=0
mylabels = ['Love','Life','God','Others']
for result in predictions:
    print(pred_data[aa])
    print(str(result[aa]) + "  -  " + mylabels[aa])
    aa +=1
    print('----------------')



# continious prediction

while True:
    user_input = input("You: ")
    if user_input == "bye":
        break
    prompt = "Quotation type: "
    myds = tf.data.Dataset.from_tensor_slices([b',,,' +  str.encode(user_input)])
    myds = myds.map(encode_mapping)
    myds = myds.padded_batch(16, padded_shapes = ([None], ()))
    
    predictions1 = loaded_model.predict(myds)
    q_result = list(predictions1[0]).index(max(predictions1[0]))
    
    print(prompt + str(mylabels[q_result]))

############ Model Development Ends #######################




