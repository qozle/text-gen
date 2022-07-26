import tensorflow as tf


##  Uses excerpts from authors:
#  C.S. Lewis "Mere christianity"
#  Franz Kaka "Metamorphasis"
#  Terry Goodkind (two books, I already forget)
#  Moby Dick


path_to_file = 'authors.txt'

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


seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)


sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

print(f"\nSequence length: {len(sequences)}\n")

# ##  look at the first five batches of 100 char sequences
# for seq in sequences.take(5):
#   print(text_from_ids(seq).numpy())


##  function to split sequences in to (input, target) pairs
def split_input_target(sequence):
	input_text = sequence[:-1]
	target_text = sequence[1:]
	return input_text, target_text

##  create dataset
dataset = sequences.map(split_input_target)

##  set batch size
BATCH_SIZE = 64

##  the actual sequence isn't shuffled, because it could be infinite
##  so part of it is shuffled, in a buffer.
BUFFER_SIZE = 10000


dataset = (
	dataset
	.shuffle(BUFFER_SIZE)
	.batch(BATCH_SIZE, drop_remainder=True)
	.prefetch(tf.data.experimental.AUTOTUNE))


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


model = tf.keras.Sequential()


model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
model.add(tf.keras.layers.LSTM(1024, stateful=True))
model.add(tf.keras.layers.Dense(vocab_size))

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss)

EPOCHS = 1

history = model.fit(dataset, epochs=EPOCHS, batch_size=1)

model.save('text-gen-model')

prediction = model.predict(["this is some test text"])

print(prediction)