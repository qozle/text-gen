import tensorflow as tf

import os
import time

#  hopefully this does garbage cleanup...?
tf.keras.backend.clear_session()

GENERATION = os.environ['gen']
EPOCHS = 5
seq_length = 100
path_to_file = 'smol.txt'
BATCH_SIZE = 64
embedding_dim = 256
rnn_units = 1024






print(f"GENERATION {GENERATION}.")

##  shakespere data- replace this

##  get the text
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

## unique characters 
vocab = sorted(set(text))

# Length of the vocabulary in chars
vocab_size = len(vocab)


##  create the StringLookup layer
##  this converts from tokens (chars) to character IDs (ints)
##  to invert his later, use invert=True
##  use get_vocabulary() method instead of sorted() above
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)


##  invert
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


##  we'll need this at some point later 
def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


##  convert the 'text' vector into a stream of character indices from our vocab
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))


ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


##  look at the first ten characters of the dataset
# for ids in ids_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))


examples_per_epoch = len(text)//(seq_length+1)


sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

##  look at the first five batches of 100 char sequences
# for seq in sequences.take(5):
#   print(text_from_ids(seq).numpy())


##  function to split sequences in to (input, target) pairs
def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text

##  create dataset
dataset = sequences.map(split_input_target)
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# print(dataset)

##  set batch size

##  the actual sequence isn't shuffled, because it could be infinite
##  so part of it is shuffled, in a buffer.
BUFFER_SIZE = 10000


##  Why are we shuffling the data?
##  I think we're shuffling the batches, not the data
dataset = (
	dataset
	.shuffle(BUFFER_SIZE)
	.batch(BATCH_SIZE, drop_remainder=True)
	.prefetch(tf.data.experimental.AUTOTUNE))





model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
model.add(tf.keras.layers.Dropout(.1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences=True)))
model.add(tf.keras.layers.LSTM(512))
model.add(tf.keras.layers.Dense(vocab_size/2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(vocab_size))



loss = tf.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss)

model.summary()


history = model.fit(dataset, epochs=EPOCHS, batch_size=64)

cue = tf.constant(['So you like prompts, eh?'])

gen_text = [cue]

for i in range(1000):
	text = model.predict(gen_text)
	gen_text.append(text)



# model.save_weights('gen_' + GENERATION + '/maxbot_gen_smol_lstm', save_format='tf')


# start = time.time()
# f_states = None
# f_cell_states = None
# b_states = None
# b_cell_states = None
# next_char = tf.constant(["How's this for a prompt, hmm?"])
# result = [next_char]



# for n in range(10000):
#   next_char, f_states, f_cell_states, b_states, b_cell_states = one_step_model.generate_one_step(inputs=next_char, f_states=f_states, f_cell_states=f_cell_states, b_states=b_states, b_cell_states=b_cell_states, return_state=True)
#   result.append(next_char)


# result = tf.strings.join(result)
# end = time.time()
# print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
# print('\nRun time:', end - start)

