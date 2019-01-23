

import collections
import random
import math
import os.path

import numpy as np
import tensorflow as tf
import gensim
import PyPDF2

import json



def bookSelector():
   books = {1: ("Frankenstein", "./books/frankenstein.pdf"), 
            2: ("The Hound of the Baskervilles", "./books/houndOfBaskervilles.pdf"),
            3: ("Moby Dick", "./books/mobyDick.pdf"),
            4: ("Tale of Two Cities", "./books/taleOfTwoCities.pdf")}
   for index in books:
      print("Select ({}) to run book2Vec on '{}'".format(index, books[index][0]))
   choice = input("Choice: ")
   while choice not in ('1', '2', '3', '4'):
      choice = input("Invalid choice. Please choose again: ")
   return books[int(choice)][1]


            # ----------- PDF Reading Functions ---------------



def readIn(pathToPDF):
   corpus = open(pathToPDF, 'rb')
   corpus = PyPDF2.PdfFileReader(corpus)
   numPages = corpus.numPages
   sentences = []
   for i in range(1, numPages):
      page = corpus.getPage(i).extractText()
      pageSentences = pageToSentences(page)
      sentences += wordify(pageSentences)
      if i%2 ==0: print(math.floor(i*100/numPages), "% Read...")
   return sentences


def pageToSentences(page):
   blackList = (';', ':', ',' , '‚', ',', '™', 'F˜˚˛˝˙˛ˆˇ˙˘˛', 'F˜˙˙', 'L˜˚˚˜˛', '˙B˝ˆ', '˙B˝' 'F˜˙˙', 'B˝', '˚ˇ', 'P˚˛˙ˇ')#, '˜', '˚', '˙', 'ˆ', 'ˇ', '˘') 
   for term in blackList:
      page = page.replace(term, '')
   page = page.lower()
   page = page.replace('\n-', '')
   page = page.replace('\n', ' ')
   page = page.replace('  ', ' ')
   page = page.replace('   ', ' ')

   sentences = page.split('.' or '!' or '?')
   return sentences


def wordify(strSentences):
   sentences = []
   for sentence in strSentences:
      sentence = sentence.split(' ')
      sentences.append(sentence)
   return sentences



def runQuerier(model):
   vocab = model.wv.vocab.keys()
   print("\nNow in 'synonym' Mode ")
   curQuery = ''
   while curQuery != "QUIT":
      curQuery = input("\nInput a Word to see its synonyms: ")
      if curQuery in vocab:
         print(model.wv.most_similar(positive = [curQuery], topn = 15))
      else: print("The word you Input was not in the book's vocabulary, try again!")
   return
'''

def enterAssistant(model):
   userString = ''
   while (keypress != escape asci):
      if keypress == alphabetic or keywpress == space:
         userString += keypress
         updateUI(userString)



def updateUI(userString):
   lastChar = userString[-1]
   lastWord = getLastWord(userString) #maybe just keep a running second string of the most recent word...
   if lastChar = (" "):
      mostSimilar = model.wv.most_similar(positive = [lastWord])[0]
      if mostSimilar[1] >= 0.8:
         userString = userString[:-1] + mostSimilar[0] + userString[-1] #some other more efficient insert equation. 
   return userString
'''


def startGensim():
   windowSize = 3
   embeddingSize = 100
   pathToPDF = bookSelector()
   sentences = readIn(pathToPDF)
   model = gensim.models.Word2Vec(sentences,
                                 window = windowSize,
                                 size = embeddingSize,
                                 sg = 1,
                                 hs = 0,
                                 negative = 5)
   model.train(sentences, 
               total_examples = len(sentences),
               epochs = 3)
   print("Model for ", pathToPDF[8:-4], " Has Been Trained!")
   runQuerier(model)
   #print(model.wv.most_similar(positive = ['captain'], topn = 5))



startGensim()




"""

#   ------ Initializing Word Pairs, and Book Vocabulary -------------


# Takes in a list of sentences. Breaks them down into pairs, and a vocabulary.


def getPairsAndVocab(sentences, windowSize, vocabSize):
   pairs, vocab = [], []
   for sentence in sentences:
      words = sentence.split(' ') #obtain the words in sentence
      vocab += words              #update vocab
      pairs += getPairs(words, windowSize)
   vocab = collections.Counter(vocab).most_common(vocabSize-1)  #Shrink vocab to top occuring words.
   # (The "-1" above is so that the vocab does not overshoot it's index number (ie try to index 3000 on a vocab of size 3000))
   return pairs, vocab




def getPairs(wordList, windowSize):
   pairs = []
   for i in range(len(wordList)):
      word = wordList[i]
      for wS in range(1, windowSize+1):
         if i - wS >= 0:
            pairs.append((word, wordList[i-wS]))
         if i + wS < len(wordList):
            pairs.append((word, wordList[i+wS]))
   return pairs






   #  ---------- Map Reduce Functions (In order to create OneHot Vectors) ----------

# Map reduces the vocabulary
def mapReduce(vocab):
   wordDict = {'UNK': 0}
   for term in vocab:
      word = term[0]
      wordDict[word] = len(wordDict)
   return wordDict


# Map reduces the pairs.
def pairReduce(initPairs, wordDict):
   pairs = []
   for pair in initPairs:
      if pair[0] not in wordDict:
         index1 = 0
      else: index1 = wordDict[pair[0]]
      if pair[1] not in wordDict:
         index2 = 0
      else: index2 = wordDict[pair[1]]
      pairs.append((index1, index2))
   return pairs






   # -------------- Batch Creation (oneHot Form) Functions --------------



def createBatch(batchSize, pairs, autoPair = False):
   batchIn = []
   batchOut = []
   i = 0
   while i < batchSize:
      randI = random.randint(1, len(pairs)-1)
      batchIn.append(pairs[randI][0])
      batchOut.append(pairs[randI][1])
      i+=1
   if autoPair != False: # Allows the potential of ensuring that certain pairs are seen at least once in training steps.
      batchIn.append(autoPair[0])
      batchOut.append(autoPair[1])
   batchIn, batchOut = np.asarray(batchIn), np.asarray(batchOut)
   return batchIn, batchOut


# Not used.... TensorFlow's tf.embedding_lookup handles what oneHots would handle
def batchToOneHot(batch, vocabSize):
   trainX, trainY = [], []
   inputs, outputs = batch[0], batch[1]
   for i in range(len(inputs)):
      oneHotIn = indexToOneHot(inputs[i], vocabSize)
      oneHotOut = indexToOneHot(outputs[i], vocabSize) #index in the vocabulary
      trainX.append(oneHotIn)
      trainY.append(oneHotOut)
   return trainX, trainY

def indexToOneHot(wordIndex, vocabSize):
   oneHot = [0]* vocabSize
   oneHot[wordIndex] = 1
   return oneHot



# ------------- Saving and Retrieving Data Functions ---------

def askAboutRestore():
   restoreModel = ''
   while restoreModel != 'y' and restoreModel != 'n':
      restoreModel = input("Previous Model Found. Would you like to restore? (y/n)")
   if restoreModel == 'y':
      restoreModel = True
   else: restoreModel = False
   return restoreModel 


def saveWordDictsToJSON(pathToPDF, wordDict, inverseDict):
   print('HERE!')
   saveTo = "./savedModels/" + pathToPDF[6:-4] + ".json"
   with open(saveTo, 'w') as outfile:
      json.dump(wordDict, outfile)
   return

def retrieveSavedWordDicts(pathToPDF):
   filePath = "./savedModels/" + pathToPDF[6:-4] + ".json"
   with open(filePath) as json_file:  
         wordDict = json.load(json_file)
   inverseDict = dict(zip(wordDict.values(), wordDict.keys()))
   return wordDict, inverseDict

# -------- The main function. Reads in the PDF book, trains the model, asks user input after --------

#Note: Take a lot of this out of a function. & Allow for things to be skipped. 


def main(windowSize = 5, vocabSize = 3000, batchSize = 600, embeddingSize = 50, validationSize = 50):

   pathToPDF = bookSelector()

# ---------- Check for a previously saved Model -------------
   restoreModel = False
   if os.path.isfile("./savedModels/" + pathToPDF[6:-4] + ".ckpt.index") and os.path.isfile("./savedModels/" + pathToPDF[6:-4] + ".json"):
      restoreModel = askAboutRestore()

   if restoreModel == True:
      wordDict, inverseDict = retrieveSavedWordDicts(pathToPDF)
      

   else: 
   # ------- Obtaining the initial Book Information -----------


      sentences = readIn(pathToPDF)
      wordPairs, vocab = getPairsAndVocab(sentences, windowSize, vocabSize)

      wordDict = mapReduce(vocab) # A dictionary mapping all top-occuring-terms to an integer {word : int}
      inverseDict = dict(zip(wordDict.values(), wordDict.keys())) #The inverse dict: {int : word}

      reducedPairs = pairReduce(wordPairs, wordDict) 

      validationX, validationY = createBatch(validationSize, reducedPairs) 

      #print("Vocab: ", vocab[20:40] , "...")       
      #print("Pairs: ", wordPairs[100:120], "...")
   


# ---------- Initializing the Model ----------------

   book1Graph = tf.Graph()


   with book1Graph.as_default():

      with tf.name_scope('data'):
         trainX = tf.placeholder(tf.int32, shape = (batchSize), name = 'centerWords') #Tensor to be fed with a series of word Indexes (each acting like a oneHot).
         trainY = tf.placeholder(tf.int32, shape = [batchSize, 1], name = 'contextWords') #Tensor to be fed with a series of word Indexes (each acting like a oneHot).
         
   # Conversion Matrix to the Hidden Layer.
      with tf.name_scope('embedding'):
         embeddingMatrix = tf.Variable(tf.random_normal([vocabSize, embeddingSize]), name = 'embeddingMatrix')

      with tf.name_scope('hiddenLayers'):
         #Built in TensorFlow function that works like oneHot matmul. No need to convert input data to oneHot.
         hiddenLayer = tf.nn.embedding_lookup(embeddingMatrix, trainX, name = 'hiddenLayer')

      with tf.name_scope('weights'):
         outputWeights = tf.Variable(tf.random_normal([vocabSize, embeddingSize]), name = 'outputWeights')
         outputBias = tf.Variable(tf.zeros([vocabSize]), name = 'outputBias')

         # Defining the Loss Function:
      with tf.name_scope('loss'):
         loss = tf.reduce_mean(tf.nn.nce_loss(
            weights = outputWeights,
            biases = outputBias,
            labels = trainY,
            inputs = hiddenLayer,
            num_sampled = 7, #look into this a little more 10 interestingly good , 25 b4
            num_classes = vocabSize), 
            name = 'loss')

      optimize = tf.train.GradientDescentOptimizer(learning_rate = 1.5).minimize(loss)


      # ------ Variables for Usage after the Training ------------


      with tf.name_scope('observedWord'):
         wordSeen = tf.placeholder(tf.int32, shape = (1), name = 'wordSeen') #The index (from wordDict) of a word seen after training.


      # A Row of 'similarity' matrix corresponds to the dot products of a single validation embedding with all other embedding vectors.
      # (Use this. Pass a given word into this. Find the words with the highest similarity values. These are the authors contextually similar words.)
      
      with tf.name_scope('obsOperations'):
         L2ofWeights = tf.sqrt(tf.reduce_sum(tf.square(outputWeights), 1, keepdims = True))
         normalizedWeights = outputWeights / L2ofWeights
         wordSeenWeights = tf.nn.embedding_lookup(normalizedWeights, wordSeen) #Extract normalized validation set embeddings.
         similarityVectors = tf.matmul(wordSeenWeights, tf.transpose(normalizedWeights))

      with tf.name_scope('similarityFromEmbeds'):
         #contextSeen = tf.placeholder(tf.float32, shape = [1, vocabSize]) # A __ hot vector that has 1 values at the positions of contet words seen.
         L2ofEmbeds = tf.sqrt(tf.reduce_sum(tf.square(embeddingMatrix), 1, keepdims = True))
         normalizedEmbeddings = embeddingMatrix / L2ofEmbeds
         wordEmbeddingWeights = tf.nn.embedding_lookup(normalizedEmbeddings, wordSeen)
         similarityVectors2 = tf.matmul(wordEmbeddingWeights, tf.transpose(normalizedEmbeddings))
         # Now find vectors with similar embedding...
         

      init = tf.global_variables_initializer()
      saver = tf.train.Saver()


      # -------- TensorFlow Sesssions -------------------

   with tf.Session(graph = book1Graph) as sess:

      init.run()
      agregateLoss = 0.0
      numSteps = 100000

      # ------ If not Restoring an OldModel: Train a New Model. --------
      if restoreModel == False: 
         print("\n Beginning Training: ")
         for step in range(numSteps):
            batch = createBatch(batchSize, reducedPairs)
            feedDict = {trainX: batch[0], trainY: batch[1].reshape((batchSize,1))}
            _, batchLoss = sess.run([optimize, loss],
                                      feed_dict = feedDict)
            agregateLoss += batchLoss

            if step%200 ==0 and step!=0:
               print("Step {0}, Average Loss: {1}".format(step, agregateLoss/step))
         #Save the Model
         save_path = saver.save(sess, "./savedModels/" + pathToPDF[6:-4] + ".ckpt")
         saveWordDictsToJSON(pathToPDF, wordDict, inverseDict)
         print("Model saved in path: %s" % save_path)

      else: #Restore Old Model
         saver.restore(sess, "./savedModels/" + pathToPDF[6:-4] + ".ckpt")
         print("Your Previous Model Has Been Restored.")

      # After Training Analysis
      seenWords = input('Type a sentence: ')
      #cleaning out sentence
      blackList = ('.', ';', ':', ',', '!')
      for item in blackList:
         seenWords = seenWords.replace(item, '')
      seenWords = seenWords.split(' ')
      #The print output
      for newWord in seenWords:
         newWord = newWord.lower()
         if newWord not in wordDict: 
            newInt = 0
         else:
            newInt = wordDict[newWord]
         print(newWord, newInt)
      #Call a session and retrieve output
         simVecs = sess.run([similarityVectors], feed_dict = {wordSeen: [newInt]})
         simVecs = simVecs[0][0]
         print(simVecs)
         #maxi = np.maximum(simVecs)
         maxIndex = np.argmax(simVecs)
         maxis = np.argpartition(simVecs, -8)[-8:]
         acctualValue = simVecs[newInt]
         print("MaxIndex: ", maxIndex, " , which corresponds to ", inverseDict[maxIndex])
         print("Acctual Index: ", acctualValue)
         for otherI in maxis:
            print("Contextually Similar Word: ", inverseDict[otherI] )
         print('\n')

       
   return

      

#main('taleOfTwoCities.pdf')
#main('mobyDick.pdf')
#main('houdOfBaskervilles.pdf')

main()



"""



