from tensorflow.keras import layers, Sequential, Input, Model
from tensorflow import reshape

import util
import config


class Encoder:
    def __init__(self):
        self.lstm = layers.RNN(layers.LSTMCell(
            config.CONTEXT_UNITS), return_state=True)
        sample_input = Input(shape=(1, util.TOKENS_LEN))
        sample_output = self.lstm(sample_input)
        # Sample model is used to save weights
        self.sample_model = Model(sample_input, sample_output)

    def predict(self, x):
        prediction, *state = self.lstm(x)

        return prediction, state

    def get_trainable_weights(self):
        return self.lstm.trainable_weights

    def save_weights(self, path):
        self.sample_model.save_weights(path)

    def load_weights(self, path):
        self.sample_model.load_weights(path)


class Decoder:
    def __init__(self):
        self.lstm = layers.RNN(layers.LSTMCell(
            config.CONTEXT_UNITS), return_sequences=True, return_state=True)
        sample_input = Input(shape=(1, util.TOKENS_LEN))
        sample_output = self.lstm(sample_input)
        # Sample model is used to save weights
        self.sample_model = Model(sample_input, sample_output)

    def predict(self, x, states):
        prediction = self.lstm(x, initial_state=states)

        return prediction

    def get_trainable_weights(self):
        return self.lstm.trainable_weights

    def save_weights(self, path):
        self.sample_model.save_weights(path)

    def load_weights(self, path):
        self.sample_model.load_weights(path)


class LLM:
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.char_model = Sequential([layers.Input(shape=(config.CONTEXT_UNITS,)), layers.Dense(
            util.TOKENS_LEN, activation='softmax')])

    # x is a tensor with shape (time_steps, features)
    def predict_in_training(self, x, y):
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

    def predict(self, x, output_length):
        x = reshape(x, (1, -1, util.TOKENS_LEN))
        y = util.get_token_vector(
            util.BOS)
        outputs = []

        _, encoder_state = self.encoder.predict(x)

        current_state = encoder_state
        for i in range(output_length):
            y = reshape(y, (1, -1, util.TOKENS_LEN))
            decoder_prediction, * \
                decoder_state = self.decoder.predict(y, current_state)
            decoder_prediction = reshape(
                decoder_prediction, (-1, config.CONTEXT_UNITS)
            )
            char_prediction = self.char_model(
                decoder_prediction)
            char_prediction = reshape(char_prediction, (util.TOKENS_LEN))

            current_state = decoder_state
            y = char_prediction
            outputs.append(char_prediction)

        return outputs

    def get_trainable_weights(self):
        encoder_weights = self.encoder.get_trainable_weights()
        decoder_weights = self.decoder.get_trainable_weights()
        char_weights = self.char_model.trainable_weights

        return encoder_weights + decoder_weights + char_weights

    def save_weights(self, path):
        self.encoder.save_weights(path + '/encoder.h5')
        self.decoder.save_weights(path + '/decoder.h5')
        self.char_model.save_weights(path + '/char_model.h5')

    def load_weights(self, path):
        self.encoder.load_weights(path + '/encoder.h5')
        self.decoder.load_weights(path + '/decoder.h5')
        self.char_model.load_weights(path + '/char_model.h5')
