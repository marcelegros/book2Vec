
   -------- Book2Vec -------------


This project seeks to learn word vectors from books. These vectors can then be used to recommend "author specific sysnonyms" for a given word, as well as draw insights from different authors semantic patterns. 


Current Structure: 

   - book2Vec-Gensim.py: Most current and efficient model. Work in progress. 
   - book2Vec(OlderTensorflowModel).py: Older Model. More granularly developed/ customized, but no longer in use for efficiency reasons.

    ---------------------------

The Structure of book2Vec-Gensim.py

   - The main() function prompts you to choose a book to use.
   - The PDF is read, broken down into sentences, and then to words.
   - A "gensim" word2Vec model is then declared. The model uses a "Skip Gram" model.
      Summary of Skip Gram: From a corpus of sequential words, a "center word", and "context word" from the text surrounding it (within specified window size) are chosen as Input and Output. The input is run through a 3 layer Neural Net. The hidden layer is a matrix of size "Vocabulary X embeddingSize".
      This middle layer learns the words contextual vector representation, within the scope of the corpus. This vector is of "embeddingSize" length.
      This embedding vector is multiplied by an "outputWeight Matrix" of size "embeddingSize X Vocabulary". This produces the dot products of the input word with the "context representation" of each word in the vocab. This dot product vecotor is used to predict the context word. 
   - The skipGram model is trained using Gradient Desent on a SoftMax output. Negative Sampling is used for efficiency.
   - The output is then used for various user interface functions. A word's VectorRepresentation can be extracted through "model.wv[word]"
   
      ---------------------------

The Structure of book2Vec(OlderTensorflowModel).py

   Here, a similar model is built manually with Tensorflow tensors and vectors. This barebones approach to the model allows for maximum customization/ understanding of information flow durring and after training.

   The model is replaced by the Gensim model for training efficiency and optimization. 
   This Tensorflow file autosaves a model and vocabulary after training (the option is given to load or retrain on the next run). 1 pretrained model of Moby Dick is included in the files.
   Execute main() to interact.


   I hope for these tools to develop into a "tonal writing assitant."
   For example, downloading Moby Dick would lead to a certain specific tonal writing aid, different from another download.



Libraries Used: 

collections (import collections)
random (import random)
math (import math)
os (import os.path)
json (import json)

numpy as np
tensorflow as tf
PyPDF2 (pip install PyPDF2)





