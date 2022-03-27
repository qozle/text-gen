import tensorflow as tf
import time
# import numpy as np
import os

#  hopefully this does garbage cleanup...?
tf.keras.backend.clear_session()



GENERATION = os.environ['gen']
EPOCHS = 3
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



##  function to split sequences in to (input, target) pairs
def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text

##  create dataset
dataset = sequences.map(split_input_target)



##  the actual sequence isn't shuffled, because it could be infinite
##  so part of it is shuffled, in a buffer.
BUFFER_SIZE = 10000


##  Why are we shuffling the data?
##  I think we're shuffling the batches, not the data
dataset = (
	dataset
	# .shuffle(BUFFER_SIZE)
	.batch(BATCH_SIZE, drop_remainder=True)
	.prefetch(tf.data.experimental.AUTOTUNE))


# Length of the vocabulary in chars
vocab_size = len(vocab)



##  define the model
class MyModel(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, rnn_units):
		super().__init__(self)
		self.rnn_units = rnn_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.dropout = tf.keras.layers.Dropout(.1)
		self.lstm1 = tf.keras.layers.LSTM(rnn_units, return_state=True, return_sequences=True)
		self.dropout = tf.keras.layers.Dropout(.1)
		self.lstm2 = tf.keras.layers.LSTM(rnn_units, return_state=True, return_sequences=True)
		self.dropout = tf.keras.layers.Dropout(.1)
		self.pre_dense = tf.keras.layers.Dense(126)
		self.dense = tf.keras.layers.Dense(vocab_size)

	def call(self, inputs, f_states=None, f_cell_states=None, f_states2=None, f_cell_states2=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x, training=training)
		x = self.dropout(x)

		##  LSTM 1 feedback
		if f_states is None:
			if training is False:
				x, f_states, f_cell_states = self.lstm1(x, training=training)
			else:
				# f_states, f_cell_states, b_states, b_cell_states = self.bidirectional.get_initial_state(x)
				initial = [tf.zeros((BATCH_SIZE, self.rnn_units)) for i in range(2)]
				x, f_states, f_cell_states = self.lstm1(x, initial_state=initial, training=training)
		else:
			x, f_states, f_cell_states = self.lstm1(x, initial_state=[f_states, f_cell_states], training=training)


		##  LSTM 2 feedback
		if f_states2 is None:
			if training is False:
				x, f_states2, f_cell_states2 = self.lstm2(x, training=training)
			else:
				# f_states, f_cell_states, b_states, b_cell_states = self.bidirectional.get_initial_state(x)
				initial = [tf.zeros((BATCH_SIZE, self.rnn_units)) for i in range(2)]
				x, f_states2, f_cell_states2 = self.lstm2(x, initial_state=initial, training=training)
		else:
			x, f_states2, f_cell_states2 = self.lstm2(x, initial_state=[f_states2, f_cell_states2], training=training)
		
		x = self.pre_dense(x, training=training)
		x = self.dense(x, training=training)

		if return_state:
			return x, f_states, f_cell_states, f_states2, f_cell_states2
		else:
			return x







#####  PREDICT  #####
class OneStep(tf.keras.Model):
	def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
		super().__init__()
		self.temperature = temperature
		self.model = model
		self.chars_from_ids = chars_from_ids
		self.ids_from_chars = ids_from_chars

		# Create a mask to prevent "[UNK]" from being generated.
		skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
		sparse_mask = tf.SparseTensor(
			# Put a -inf at each bad index.
			values=[-float('inf')]*len(skip_ids),
			indices=skip_ids,
			# Match the shape to the vocabulary
			dense_shape=[len(ids_from_chars.get_vocabulary())])
		self.prediction_mask = tf.sparse.to_dense(sparse_mask)

	@tf.function
	def generate_one_step(self, inputs, f_states=None, f_cell_states=None, f_states2=None, f_cell_states2=None, return_state=True):
		# Convert strings to token IDs.
		input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
		input_ids = self.ids_from_chars(input_chars).to_tensor()

		# Run the model.
		# predicted_logits.shape is [batch, char, next_char_logits]
		predicted_logits, f_states, f_cell_states, f_states2, f_cell_states2= self.model(inputs=input_ids, f_states=f_states, f_cell_states=f_cell_states, f_states2=f_states2, f_cell_states2=f_cell_states2, return_state=return_state)
		# Only use the last prediction.
		predicted_logits = predicted_logits[:, -1, :]
		predicted_logits = predicted_logits/self.temperature
		# Apply the prediction mask: prevent "[UNK]" from being generated.
		predicted_logits = predicted_logits + self.prediction_mask

		# Sample the output logits to generate token IDs.
		predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
		predicted_ids = tf.squeeze(predicted_ids, axis=-1)

		# Convert from token ids to characters
		predicted_chars = self.chars_from_ids(predicted_ids)

		# Return the characters and model state.
		return predicted_chars, f_states, f_cell_states, f_states2, f_cell_states2


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# model = tf.saved_model.load('maxbot_gen4_smol_lstm')

model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)


model.compile(optimizer='adam', loss=loss)

for input, target in dataset.take(1):
	model.train_on_batch(input, target)



model.load_weights('gen_' + GENERATION + '/maxbot_gen_smol_lstm')


one_step_model = OneStep(model, chars_from_ids, ids_from_chars)






start = time.time()
f_states = None
f_cell_states = None
f_states2 = None
f_cell_states2 = None
next_char = tf.constant(["I heard you like prompts, bro."])
result = [next_char]



for n in range(1000):
  next_char, f_states, f_cell_states, f_states2, f_cell_states2 = one_step_model.generate_one_step(next_char, f_states=f_states, f_cell_states=f_cell_states, f_states2=f_states2, f_cell_states2=f_cell_states2, return_state=True)
  result.append(next_char)



result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
