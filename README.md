This is a machine learning model made using tensorflow that can be trained to emulate a writing style of a body of text.  

It is sloppy as hell and uses tensorflow.  Eventually I would like to integrate this into my dream-tweeter as it currently uses a pre-trained (unsupervised...?!) model by DeepAI.  

`build-data-lstm.py` trains a model and saves the best weights.  `test.py` will load the weights and make predictions using the model.  

This was taken from a tutorial on tensorflow's site (https://www.tensorflow.org/text/tutorials/text_generation) and used mostly for learning purposes.  I recognize there are other approaches to doing this that I could do (predicting entire words), as well as that it may be sloppy to have to load the model architecture twice (in build-data-lstm, AND test.py).  I also know it could be made more efficient and sped up.  

Something I've thought of doing with this is scraping text from wikipedia or taking submissions (probably not the best idea) and retraining the model every night with new data.  The text file could be given a character limit, and cycle out old content (too large of a file would be useless anyway as I'm pretty sure an LSTM can still only remember so far back).  

Eventually I will refactor this.  Note that I removed the body of text that the model trains on.  Also note that the model weights that I trained aren't available on this git because the size is too large.  You can find them at:
	- https://01014.org/ml_weights/gen_6/checkpoint
	- https://01014.org/ml_weights/gen_6/maxbot_gen_smol_lstm.data-00000-of-00001
	- https://01014.org/ml_weights/gen_6/maxbot_gen_smol_lstm.index

For the script to work, these three files should be saved in a folder called `gen_6` in the top directory that the scripts are contained in.