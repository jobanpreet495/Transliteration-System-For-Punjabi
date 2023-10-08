from keras.models import Model
from keras.layers import Input, LSTM, Dense, RNN
from tensorflow import keras
import numpy as np
import pickle

model = keras.models.load_model("s2s")

with open("input_token_index.pkl", "rb") as file:
    input_token_index = pickle.load(file)

with open("target_token_index.pkl", "rb") as file:
    target_token_index = pickle.load(file)

num_encoder_tokens=len(input_token_index)
num_decoder_tokens=len(target_token_index)
latent_dim=256
max_encoder_seq_length=25
max_decoder_seq_length=24


encoder_inputs = model.input[0]  # input_1
encoder_lstm_1 = model.layers[2]
encoder_outputs_1, h1, c1 = encoder_lstm_1(encoder_inputs)
encoder_lstm_2 = model.layers[4]
encoder_outputs, h2, c2 = encoder_lstm_2(encoder_outputs_1)
encoder_states = [h1, c1, h2, c2]
decoder_inputs=model.input[1]

out_layer1 = model.layers[3]
out_layer2 = model.layers[5]


decoder_dense = model.layers[6]

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input_h1 = Input(shape=(latent_dim,))
decoder_state_input_c1 = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c,
                         decoder_state_input_h1, decoder_state_input_c1]
d_o, state_h, state_c = out_layer1(
    decoder_inputs, initial_state=decoder_states_inputs[:2])
d_o, state_h1, state_c1 = out_layer2(
    d_o, initial_state=decoder_states_inputs[-2:])
decoder_states = [state_h, state_c, state_h1, state_c1]
decoder_outputs = decoder_dense(d_o)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c, h1, c1 = decoder_model.predict(
            [target_seq] + states_value) #######NOTICE THE ADDITIONAL HIDDEN STATES

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c, h1, c1]

    return decoded_sentence





# # def predict(s):
# s=["joban"]
# encoder_input_data = np.zeros(
# (len(s), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
# )

# for i, input_text in enumerate(s):
#     for t, char in enumerate(input_text):
#         encoder_input_data[i, t, input_token_index[char]] = 1.0
#     encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0

# decoded_sentences = []
# for input_data in encoder_input_data:
#     decoded_sentence = decode_sequence(input_data[np.newaxis, :, :])
#     decoded_sentences.append(decoded_sentence)
# print(decoded_sentences)

