
   -------- Book2Vec -------------


This project seeks to learn word vectors from books. These vectors can then be used to recommend "author specific sysnonyms" for a given word, as well as draw insights from different authors semantic patterns. 
(See bottom of this file for Libraries Required / Dependencies)

Current Structure: 

   - book2Vec-Gensim.py: Most current and efficient model. Work in progress. 
   - book2Vec(OlderTensorflowModel).py: Older Model. More granularly developed/ customized, but no longer in use for efficiency reasons.

    ---------------------------

The Structure of book2Vec-Gensim.py

   - The main() function prompts you to choose a book to use.
   - The PDF is read, broken down into sentences, and then to words.
   - A "gensim" word2Vec model is then declared. The model uses a "Skip Gram" model.

   - After training, the model enters 'synonym' mode. The user can enter a word, in exchange for similar words learned from the context of the author's writing. 
   - The user may also enter a 'Word Equation,' to retrieve the vectors most similar to the addition and subtraction of words. For example: 'King - Man + Woman ' has been known to return 'Queen.' 
   - Manually, in the code, a word's VectorRepresentation can be extracted through "model.wv[word]"


    ---------------------------


Summary of Skip Gram Model:

      The model uses a 'center word' as input,and 'context word' as output. The model randomly samples the corpus, and learns a model that makes the observation of the novel's context words the most likely, given it's center words. 
      To achieve this, the input is run through a 3 layer Neural Net. The hidden layer is a matrix of size "Vocabulary X embeddingSize".
      Each row of this middle layer is a given word's contextual vector representation, within the scope of the corpus. This vector is of "embeddingSize" length.
      This embedding vector is multiplied by an "outputWeight Matrix" (of size "embeddingSize X Vocabulary"). This produces the dot products of the input word with the "context representation" of each word in the vocab. This dot product vecotor is softMaxed, and used to predict the 'most likely' context word. The model changes itself to improve these predictions.

     The skipGram model is trained using Gradient Desent on a SoftMax output. Negative Sampling is used for efficiency.
   
      ---------------------------

The Structure of book2Vec(OlderTensorflowModel).py

   Here, a similar model is built manually with Tensorflow tensors and vectors. This barebones approach to the model allows for maximum customization/ understanding of information flow durring and after training.

   The model is replaced by the Gensim model for training efficiency and optimization. 
   This Tensorflow file autosaves a model and vocabulary after training (the option is given to load or retrain on the next run). 1 pretrained model of Moby Dick is included in the files.
   Execute main() to interact.


   I hope for these tools to develop into a "tonal writing assitant."
   For example, downloading Moby Dick would lead to a certain specific tonal writing aid, different from another download.



Libraries Used/ Dependencies: 

PyPDF2 (pip install PyPDF2)
gensim (pip install gensim)
numpy as np
tensorflow as tf

copy (import copy)
collections (import collections)
random (import random)
math (import math)
os (import os.path)
json (import json)






