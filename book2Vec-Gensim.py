

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






