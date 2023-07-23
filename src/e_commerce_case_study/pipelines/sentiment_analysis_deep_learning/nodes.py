import os
import yaml
import pickle
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

log = logging.getLogger(__name__)

def model_dl():
    input_shape = (yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['max_length'])
    vocab_size = yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['vocab_size']
    embedding_dim = yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['embedding_dim']
    lstm_neurons = yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['lstm_neurons']
    dense_neurons = yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['dense_neurons']
    dropout_rate = yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['dropout']

    
    input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=input_shape)(input)
    x = tf.keras.layers.LSTM(lstm_neurons, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(lstm_neurons)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(dense_neurons, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(dense_neurons // 2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model



def train_dl_model(x_train, y_train, x_test, y_test):
    try:
        x_train = pickle.load(open("data/05_model_input/x_train.pkl", "rb"))
        y_train = pickle.load(open("data/05_model_input/y_train.pkl", "rb"))
        x_test = pickle.load(open("data/05_model_input/x_test.pkl", "rb"))
        y_test = pickle.load(open("data/05_model_input/y_test.pkl", "rb"))
    except Exception as err:
        log.error("Error occured while loading data: %s", err)
    finally:
        log.info("Succesfully loded the data")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['monitor'],
                                                      patience=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['early_stopping'],
                                                      verbose= yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['verbose'],
                                                      restore_best_weights=True)
    
    reduce_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['monitor'],
                                                          factor=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['factor'],
                                                          mode=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['mode'],
                                                          patience=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['patience'],
                                                          min_lr=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['min_lr'])
    
    model = model_dl()

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer = tf.keras.optimizers.SGD(learning_rate=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['lr_rate'],
                                                      momentum=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['momentum']),
                  metrics=['accuracy'])
    
    log.info(model.summary())

    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)


    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    history = model.fit(x = x_train,
                    y = y_train, 
                    validation_data=(x_test, y_test),
                    epochs=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['epochs'],
                    batch_size=yaml.safe_load(open('conf/base/parameters/sentiment_analysis_deep_learning.yml'))['params']['batch_size'],
                    callbacks= [early_stopping, reduce_plateau])
    
    log.info(history)

    