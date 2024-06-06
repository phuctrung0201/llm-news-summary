from tensorflow import zeros, concat
from tensorflow.math import argmax

EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'
SEP = ' '


def initToken():
    tokens = [
        EOS,
        BOS,
        SEP,
        UNK
    ]

    for i in range(65, 91):
        tokens.append(chr(i))
        tokens.append(chr(i + 32))

    return tokens


TOKENS = initToken()

TOKENS_LEN = len(TOKENS)


def get_token(char):
    for i in range(TOKENS_LEN):
        if TOKENS[i] == char:
            return i

    return 3


def get_token_vector(char):
    token = get_token(char)

    vector = zeros(TOKENS_LEN, 'float32')

    return concat([vector[:token], [1.0], vector[token + 1:]], 0)


def get_char(token):
    char_index = argmax(token)
    return TOKENS[char_index]


def get_text(token):
    if token == 0:
        return EOS

    if token == 1:
        return BOS

    if token == 2:
        return ' '

    if token < 4 or token > 61:
        return UNK

    return chr(token + 61)


def get_sentence_vector(sen):
    vector = [get_token_vector(BOS)]
    for c in sen:
        vector = concat([vector, [get_token_vector(c)]], 0)
    vector = concat([vector, [get_token_vector(EOS)]], 0)

    return vector
