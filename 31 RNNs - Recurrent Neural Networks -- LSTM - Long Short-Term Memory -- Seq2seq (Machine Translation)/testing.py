import numpy as np
from preprocessing import target_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, vectorize
from training import encoder_inputs, decoder_inputs, encoder_states, decoder_lstm, decoder_dense, encoder_input_data, num_decoder_tokens, latent_dim
from tensorflow import keras
from keras.layers import Input
from keras.models import Model, load_model

# For encoder, reuse our training model's input and output states for our testing model
training_model = load_model('training_model.keras')
encoder_inputs = training_model.input[0]  # input layer
_encoder_outputs, state_hidden_enc, state_cell_enc = training_model.layers[2].output  # hidden LSTM layer
encoder_states = [state_hidden_enc, state_cell_enc]

encoder_model = Model(encoder_inputs, encoder_states)

# For decoder, we don't use teacher forcing anymore, so input states' shape is equal to our LSTM layer size - so we can re-init them with output states
decoder_state_input_hidden = Input(shape=(latent_dim,)) # input layer
decoder_state_input_cell = Input(shape=(latent_dim,)) # input layer
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs) # hidden LSTM layer
decoder_states = [state_hidden, state_cell]

decoder_outputs = decoder_dense(decoder_outputs) # output dense layer

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Translates an input sentence, in a form of a one-hot tensor, into a target, human-readable sequence
def decode_sequence(test_input):
    # Initiate decoder's states by encode the input sequence as state vectors
    states_value = encoder_model.predict(test_input)

    # Initiate target sequence - the next word we try to predict in the target sentence
    # Generate an empty tensor
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate it with the start token, as the first token of the target sequence
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    decoded_sentence = ''

    # Sampling loop (to simplify, we assume a batch of sequences of size 1)
    stop_condition = False
    while not stop_condition:

        # Run the decoder model to get all possible output tokens (with probabilities) & states
        output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

        # Choose the token with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]

        decoded_sentence += " " + sampled_token

        # Exit condition: either reached the stop token or max supported sentence length
        if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Re-init the target sequence with the predicted word - to pass it as input for the next time step
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states - to also pass it as input for the next time step
        states_value = [hidden_state, cell_state]

    return decoded_sentence

# Training text translation
# Range is a number of test sentences to translate
for seq_index in range(10):
    test_input = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(test_input)
    print('---')
    print(f'Input sentence {seq_index}:', input_docs[seq_index])
    # print(f'Input tensor {seq_index}:', test_input)
    print(f'Decoded sentence {seq_index}:', decoded_sentence)


# New text translation
def translate(new_sentence):
    new_test_input = vectorize(new_sentence)
    new_decoded_sentence = decode_sequence(new_test_input)
    print('---')
    print(f'Input sentence:', new_sentence)
    # print(f'Input tensor:', new_test_input)
    print(f'Decoded sentence:', new_decoded_sentence)

translate("Wow! Hello!")

# Interactive translation
def translate_user_input():
    while True:
        translate(input("Please enter your text to translate: "))

translate_user_input()