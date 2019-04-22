from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np
import nltk
import csv
import time

def get_val(arr):
    return arr[1]

batch_size = 128  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 1000  # Number of samples to train on.
data_path = './data/data.csv' # Path to the data file
input_vector_size = 5000
target_vector_size = 5000
max_input_size = 100
max_output_size = 25

# Vectorize the data.
input_texts = []
target_texts = []
input_words = {}
target_words = {}
num_lines = 0
print('Reading CSV')
start = time.time()
with open(data_path, 'r') as f:
    lines = csv.reader(f)
    next(lines)
    while(num_lines < num_samples):
        try:
            row = next(lines)
            if len(row) != 2:
                continue
            target_text, input_text = row
            target_text = 'STARTOFSENTENCE ' + target_text + ' ENDOFSENTENCE'
            inwords = nltk.word_tokenize(input_text)
            outwords = nltk.word_tokenize(target_text)
            if len(inwords) > max_input_size+2 or len(outwords) > max_output_size+2:
                continue
            
            input_texts.append(input_text)
            target_texts.append(target_text)
            for word in inwords:
                if word not in input_words:
                    input_words[word]=1
                else:
                    input_words[word]+=1
            for word in outwords:
                if word not in target_words:
                    target_words[word]=1
                else:
                    target_words[word]+=1
            num_lines+=1
        except UnicodeDecodeError:
            pass
        except StopIteration:
            break
end = time.time()
print('Reading CSV took ' + str(end-start) + ' seconds')

input_words = [[k,v] for k,v in input_words.items()]
input_words.sort(key=get_val, reverse = True)
target_words = [[k,v] for k,v in target_words.items()]
target_words.sort(key=get_val, reverse = True)
input_words = sorted(list(np.array(input_words)[:input_vector_size+2,0]))
target_words = sorted(list(np.array(target_words)[:target_vector_size+2,0]))
num_encoder_tokens = len(input_words)+1
num_decoder_tokens = len(target_words)+1
max_encoder_seq_length = max([len(nltk.word_tokenize(txt)) for txt in input_texts])
max_decoder_seq_length = max([len(nltk.word_tokenize(txt)) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
input_token_index['UNK']=len(input_token_index)
target_token_index = dict([(word, i) for i, word in enumerate(target_words)])
target_token_index['UNK']=len(target_token_index)

#Model definition
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                    initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

num_batches = num_samples//batch_size
print('Number of batches: ' + str(num_batches))
for epoch in range(epochs):
    for batch in range(num_batches):
        print('Training batch ' + str(batch) + ' in epoch ' + str(epoch))
        start = time.time()
        encoder_input_data = np.zeros((batch_size, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
        decoder_input_data = np.zeros((batch_size, max_decoder_seq_length, num_decoder_tokens),dtype='float32')
        decoder_target_data = np.zeros((batch_size, max_decoder_seq_length, num_decoder_tokens),dtype='float32')

        start_index = (batch_size*batch)
        end_index = (batch_size*(batch+1))
        for i, (input_text, target_text) in enumerate(list(zip(input_texts, target_texts))[start_index:end_index]):
            for t, word in enumerate(nltk.word_tokenize(input_text)):
                if word not in input_token_index:
                    word = 'UNK'
                encoder_input_data[i, t, input_token_index[word]] = 1.
            for t, word in enumerate(nltk.word_tokenize(target_text)):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                if word not in target_token_index:
                    word = 'UNK'
                decoder_input_data[i, t, target_token_index[word]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start word.
                    decoder_target_data[i, t - 1, target_token_index[word]] = 1.
        model.train_on_batch([encoder_input_data, decoder_input_data], decoder_target_data)
        end = time.time()
        print('Epoch ' + str(epoch) + ' batch ' + str(batch) + ' took ' + str(end-start) + ' seconds')
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict(
    (i, word) for word, i in input_token_index.items())
reverse_target_word_index = dict(
    (i, word) for word, i in target_token_index.items())

def main():
    train_encoder_input_data = get_encoder_data_batch(0, batch_size)
    test_encoder_input_data = get_encoder_data_batch(num_batches*batch_size, -1)
    print('Evaluating training samples')
    start = time.time()
    evaluate_sequences(train_encoder_input_data, 'train_examples.txt')
    end = time.time()
    print('Evaluating training samples took ' + str(end-start) + ' seconds')
    print('Evaluating testing samples')
    start = time.time()
    evaluate_sequences(test_encoder_input_data, 'test_examples.txt')
    end = time.time()
    print('Evaluating testing samples took ' + str(end-start) + ' seconds')

def get_encoder_data_batch(start_index, end_index):
    data = np.zeros((batch_size, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for i, (input_text, target_text) in enumerate(list(zip(input_texts, target_texts))[start_index:end_index]):
        for t, word in enumerate(nltk.word_tokenize(input_text)):
            if word not in input_token_index:
                word = 'UNK'
            data[i, t, input_token_index[word]] = 1.
    return data

def evaluate_sequences(data, fname):
    txt = ''
    for seq_index in range(batch_size):
        input_seq = data[seq_index]
        decoded_sentence = decode_sequence(input_seq)
        txt += 'Input sentence: ' + input_texts[seq_index]
        txt += '\n'
        txt += 'Decoded sentence: ' + decoded_sentence
        txt += '\n\n'
    f = open(fname, "w+")
    f.write(txt)
    f.close()

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first word of target sequence with the start word.
    target_seq[0, 0, target_token_index['STARTOFSENTENCE']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        if sampled_word == 'ENDOFSENTENCE':
            break
        decoded_sentence += sampled_word + ' '

        # Exit condition: hit max length
        if (len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

if(__name__=='__main__'):
	main()