from tensorflow import constant

EOS = '[EOS]'
BOS = '[BOS]'
UNK = '[UNK]'


def get_token(char):
    if char == EOS:
        return 0

    if char == BOS:
        return 1

    if char == ' ':
        return 2

    char_code = ord(char)

    if char_code < 65 or char_code > 122:
        return 3  # Unknown char

    return char_code - 61


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


def get_vectorized_sentence(sen):
    vector = [[get_token(BOS)]]
    for c in sen:
        vector.append([get_token(c)])
    vector.append([get_token(EOS)])

    return constant(vector, 'float32')
