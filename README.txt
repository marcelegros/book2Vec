
-------- Book2Vec -------------




This project seeks to learn word vectors from books. These vectors can then be used to recommend "author specific sysnonyms" for a given word, as well as draw insights from different authors semantic patterns. 

Current Structure of book2Vec.py
   - The main() function prompts you to choose a book to use.
   - The PDF is read, broken down into sentences, and then to words.
   - For each word in a sentence, word pairs are created, connect the word to its surrounding context within a certain window size.
   - All words in the framework are then mapped to numbers, so that they may be passed into the NN.
   -  The tensorFlow model is initialized: A skip-gram model, which uses a Noice-Constructive Estimation Loss. The dimensions of the hidden layer (as well as the training batchSize) are customizable through the "embeddingSize" (and "batchSize") variables.
   
	After the training is complete, another set of TensorFlow variables are available, to  extract insights from the model.
      - similarityVectors : Returns a 1D array - The dot products of the input word's normalized hidden representation (embedded components), and the normalized hidden representations of all other words. The highest values of these show the most similarity, and are picked out.
      - similarityVectors2: Very similar to the former, except the hidden representations are extracted from the outputWeight Matrix, rather than the inputEmbedding Matrix. 


Saving and Loading : 
   After the training is run on a PDF, the session and word mappings are saved. The next time the user looks to use book2Vec on the same book, they will be asked if they would like to train a new model, or reuse the model from the last training session. 
   (NOTE: At the moment, in order to load a past model, the parameters of the current training must be the same as those of the past model. This will be better adressed in the future.)


   In further developing this project, I hope to optimize the training functions for more meaningful and rapid embedding and similarity extraction. I hope for these tools to develop into a "tonal writing assitant."
   For example, you download a Herman Melville book, and have a somewhat tailored thesaurus at your disposal.



Libraries Used: 

collections (import collections)
random (import random)
math (import math)
os (import os.path)
son (import json)

numpy as np
tensorflow as tf
PyPDF2 (pip install PyPDF2)





