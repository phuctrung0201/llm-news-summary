from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow import constant, reshape

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
            config.CONTEXT_UNITS), return_sequences=True)

    def predict(self, x, states):
        prediction = self.lstm(x, initial_state=states)

        return prediction


class LLM:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        input_layer = layers.Input(shape=(config.CONTEXT_UNITS,))
        dense = layers.Dense(1, activation='softmax')(input_layer)
        self.char_model = models.Model(inputs=input_layer, outputs=dense)

    # x is a tensor with shape (batch_size, time_steps, features)
    def predict(self, x):
        bos_vector = constant(
            [[[util.get_token(util.BOS)]]], np.float32)

        _, encoder_state = self.encoder.predict(x)

        decoder_prediction = self.decoder.predict(bos_vector, encoder_state)
        decoder_prediction = reshape(
            decoder_prediction, (-1, config.CONTEXT_UNITS))

        print(decoder_prediction)
        char_prediction = self.char_model.predict(decoder_prediction)

        return char_prediction
