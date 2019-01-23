

import collections
import copy
import random
import math
import os.path

import numpy as np
import tensorflow as tf
import gensim
import PyPDF2

import json




# ------------ Function to Select a Book File -------------


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




# -------- Querier Functions ------------------




def runQuerier(model):
   vocab = model.wv.vocab.keys()
   print("\nNow in 'synonym' Mode. \n\n You may enter a single word OR\n add words together (using '+' or '-') to get words similar to the sum or differnce of their vecotrs. \n For example: king - man + woman = queen! \n\nEnter 'QUIT' To escape. ")
   curQuery = ''
   while curQuery != "QUIT":
      curQuery = input("\nInput a Word to see its synonyms: ")
      if '+' in curQuery or '-' in curQuery:
         curQuery = curQuery.replace(' ', '')
         complexQuery(curQuery, model, vocab)
      elif curQuery in vocab:
         print(model.wv.most_similar(positive = [curQuery], topn = 15))
      else: print("The word you Input was not in the book's vocabulary, try again!")
   return



def complexQuery(query, model,vocab):
   positives, negatives = [], []
   queryOperations(query, model, positives, negatives)
   for word in positives+negatives:
      if word not in vocab:
         print("One of the Words you input is not in the book's vocabulary, try again!")
         return
   print(model.wv.most_similar(positive = positives, negative = negatives, topn=15))
   return



def queryOperations(query, model, positives , negatives , startI=0, curOperation = '+'):
   i = copy.copy(startI)
   while i < len(query)-1:
      while i < len(query)-1 and query[i] != '+' and query[i] != '-':
         i += 1
      newOperation = query[i]
      if i == len(query) -1:
         appendPosOrNeg(curOperation, query[startI: i+1], positives, negatives)
      else:
         appendPosOrNeg(curOperation, query[startI: i], positives, negatives)
      curOperation = newOperation
      i = queryOperations(query, model, positives, negatives, i+1, curOperation)
   return i



def appendPosOrNeg(curOperation, word, positives, negatives):
   if curOperation == '+':
      positives.append(word)
   else: 
      negatives.append(word)
   print(word)
   #put the if:else: here. in the queryOp func, put something else that passes different words in based on "if we're at the end" or not.
   return




# ----------- THE MODEL  --------------------


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










