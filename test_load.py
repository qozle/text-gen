import tensorflow as tf
import time
# import numpy as np
import os



EPOCHS = 3
seq_length = 100
path_to_file = 'all_journals_1.txt'
BATCH_SIZE = 64
embedding_dim = 256
rnn_units = 1024


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




model = tf.keras.models.load_model('authors_model_256')



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


