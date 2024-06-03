from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow import reshape

import util
import numpy as np
import config


class Encoder:
    def __init__(self):
        self.lstm = layers.RNN(layers.LSTMCell(
            config.CONTEXT_UNITS), return_state=True)

    def predict(self, x):
        prediction, *state = self.lstm(x)

        return prediction, state


class Decoder:
    def __init__(self):
        self.lstm = layers.RNN(layers.LSTMCell(
            config.CONTEXT_UNITS), return_sequences=True, return_state=True)

    def predict(self, x, states):
        prediction = self.lstm(x, initial_state=states)

        return prediction


class LLM:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        input_layer = layers.Input(shape=(config.CONTEXT_UNITS,))
        dense = layers.Dense(
            util.TOKENS_LEN, activation='softmax')(input_layer)
        self.char_model = models.Model(inputs=input_layer, outputs=dense)

    # x is a tensor with shape (time_steps, features)
    def train_predicting(self, x, y):
        x = reshape(x, (1, -1, util.TOKENS_LEN))
        y = reshape(y, (1, -1, util.TOKENS_LEN))

        _, encoder_state = self.encoder.predict(x)

        decoder_prediction, * \
            states = self.decoder.predict(y, encoder_state)
        decoder_prediction = reshape(
            decoder_prediction, (-1, config.CONTEXT_UNITS)
        )

        char_prediction = self.char_model.predict(
            decoder_prediction, verbose=0)

        return reshape(char_prediction, (-1, util.TOKENS_LEN))
