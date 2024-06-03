from tensorflow.keras import layers, Sequential
from tensorflow import reshape

import util
import config


class Encoder:
    def __init__(self):
        self.lstm = layers.RNN(layers.LSTMCell(
            config.CONTEXT_UNITS), return_state=True)

    def predict(self, x):
        prediction, *state = self.lstm(x)

        return prediction, state

    def get_trainable_weights(self):
        return self.lstm.trainable_weights


class Decoder:
    def __init__(self):
        self.lstm = layers.RNN(layers.LSTMCell(
            config.CONTEXT_UNITS), return_sequences=True, return_state=True)

    def predict(self, x, states):
        prediction = self.lstm(x, initial_state=states)

        return prediction

    def get_trainable_weights(self):
        return self.lstm.trainable_weights


class LLM:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.char_model = Sequential([layers.Input(shape=(config.CONTEXT_UNITS,)), layers.Dense(
            util.TOKENS_LEN, activation='softmax')])

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

        char_prediction = self.char_model(
            decoder_prediction)

        return reshape(char_prediction, (-1, util.TOKENS_LEN))

    def get_trainable_weights(self):
        return self.encoder.get_trainable_weights(), self.decoder.get_trainable_weights(), self.char_model.trainable_weights
