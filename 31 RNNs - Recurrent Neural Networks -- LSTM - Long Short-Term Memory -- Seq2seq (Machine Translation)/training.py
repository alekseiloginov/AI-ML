from preprocessing import num_encoder_tokens, num_decoder_tokens, decoder_target_data, encoder_input_data, decoder_input_data, decoder_target_data
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# Choose a dimensionality - number of units of a hidden LSTM layer
latent_dim = 256

# Choose a batch size and a number of epochs:
batch_size = 50
epochs = 200

# Encoder training setup
encoder_inputs = Input(shape=(None, num_encoder_tokens))

encoder_lstm = LSTM(latent_dim, return_state=True) # we need state only - to pass to decoder
_encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Decoder training setup:
decoder_inputs = Input(shape=(None, num_decoder_tokens)) # ground truth for teacher forcing

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True) # we need outputs sequences only - to pass to output layer
decoder_outputs, _decoder_state_hidden, _decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax') # output layer returning 0-1 probabilities distribution for each token
decoder_outputs = decoder_dense(decoder_outputs)


def train_model():
    # Building the training model
    # in a functional style (input & output only) - Keras will derive all the layers from internal connections b/w inputs/outputs and layers
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print("Model summary:\n")
    training_model.summary()
    print("\n\n")
    # Compile the model:
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model:
    print("Training the model:\n")
    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size,
                       epochs=epochs, validation_split=0.2)
    training_model.save('training_model.keras')

# uncomment to train, comment out to skip training
train_model()