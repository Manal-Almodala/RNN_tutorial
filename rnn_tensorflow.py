import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf 

nltk.download("book")

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))
    
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])


# Print an training data example
x_example, y_example = X_train[17], y_train[17]
print "x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example)
print "\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example)


hidden_dim = 100
vocab_size = 8000




graph = tf.Graph()
with graph.as_default():
	#Parameters
	U = tf.Variable(tf.truncated_normal([hidden_dim, vocab_size]), -0.1, 0.1)
	W = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim]), -0.1, 0.1)
	V = tf.Variable(tf.truncated_normal([vocab_size, hidden_dim]), -0.1, 0.1)

	def forward_propagation(X):
		# input  : list of index of words
		# output : hidden state & output (probability of next words) 
		T = len(X)
		S = tf.zeros((T+1,hidden_dim))
		O = tf.zeros((T,vocab_size))
		
		for t in xrange(T):
			S[t] = tf.tanh( U[:,X[t]] + tf.matmul(W,S[t-1]) )
			O[t] = tf.nn.softmax(tf.matmul(V,S))
		return S, O

	def predict(X):
		# Returns maximum argument and output
		s, o = forward_propagation(X)
		return np.argmax(o, axis=1), o

	def calculate_total_loss(X, y):
		y_hat, o = predict(X)
		for i in xrange(y):
			loss += -np.log(o[i][y[i]]) 
		return loss 

it to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])
	
	
