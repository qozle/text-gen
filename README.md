This is a machine learning model made using tensorflow that can be trained to emulate a writing style of a body of text.  

authors.txt is a file that I compiled from excerpts from some famous authors including: C.S. Lewis, Franz Kafka, Terry Goodkind, and Herman Melville.

It is sloppy as hell and uses tensorflow.  Eventually I would like to integrate this into my dream-tweeter as it currently uses a pre-trained (unsupervised...?!) model by DeepAI.  

`build-data-lstm.py` trains and saves the model. 

This was taken (mostly) from a tutorial on tensorflow's site (https://www.tensorflow.org/text/tutorials/text_generation) and used mostly for learning purposes.  I recognize there are other approaches to doing this that I could do (predicting entire words), as well as that it may be sloppy to have to load the model architecture twice (in build-data-lstm, AND test.py).  I also know it could be made more efficient and sped up.  

Something I've thought of doing with this is scraping text from wikipedia or taking submissions (probably not the best idea) and retraining the model every night with new data.  The text file could be given a character limit, and cycle out old content (too large of a file would be useless anyway as I'm pretty sure an LSTM can still only remember so far back).  
