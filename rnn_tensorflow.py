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

#nltk.download("book")

BATCH_SIZE = 100
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

# define tensor shape
#train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
#test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


# collect batches of images before processing
#X, Y = tf.train.batch([X_train, y_train], batch_size=BATCH_SIZE)

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


#X_batch = split_list(list(X_train), 70000)
#Y_batch = split_list(list(y_train), 70000)
#X=X_batch[0]
#Y=Y_batch[0]

X=X_train
Y=y_train


import time

start = time.time()

graph = tf.Graph()
with graph.as_default():

	#X = tf.placeholder(tf.float32, [None, vocab_size])
	#Y = tf.placeholder(tf.float32, [None, vocab_size])

	#Parameters
	U = tf.Variable(tf.truncated_normal([vocab_size, hidden_dim], -0.1, 0.1)) #(8000,100)
	W = tf.Variable(tf.truncated_normal([hidden_dim, hidden_dim], -0.1, 0.1)) #(100,100)
	V = tf.Variable(tf.truncated_normal([vocab_size, hidden_dim], -0.1, 0.1)) #(8000,100)

	def forward_propagation(x):
		# We call this function for each sentence.
		# input  : list of index of words
		# output : hidden state & output (probability of next words) 
		"""
		X:
		SENTENCE_START what are n't you understanding about this ? !
		[0, 51, 27, 16, 10, 861, 54, 25, 34, 69]

		ground truth label:
		what are n't you understanding about this ? ! SENTENCE_END
		[51, 27, 16, 10, 861, 54, 25, 34, 69, 1]
		"""
		T = len(x)
		s = [tf.placeholder(tf.float32, [hidden_dim,1])] #s = tf.placeholder(tf.float32, [T+1, hidden_dim])
		o = [] #y_ = tf.placeholder(tf.float32, [T, vocab_size])
		for t in xrange(T): #[0, 51, 27, 16, 10, 861, 54, 25, 34, 69]
			embed = tf.reshape(tf.nn.embedding_lookup(U, x[t]), [hidden_dim,1] ) #(100,1)
			new_s = tf.tanh( embed + tf.matmul(W,s[t-1])) # 100*100*100*1 -> 100*1
			s.append(new_s)
			o.append(tf.matmul(V, new_s)) # I took softmax out and sent it to softmax_cross_entropy_with_logits.
			#             8000*100*100*1 = (8000,1)
		return s, o

	"""
	def predict(x):
		# For one sentence!
 		# Returns maximum argument and output
		s, o = forward_propagation(X)
		return np.argmax(o, axis=1), o # This o is softmax treated o.

	How will you translate this code into TF?
	def calculate_total_loss(X, Y):
		# Input is the list of sentences!
		loss=0
		for senten_i in xrange(len(Y)):
			y_hat_index, output_prob = predict(X[senten_i])
			# shape of output_prob = (sentence_length, vocab_size) 
			# in forward_propagation fuction: o = tf.zeros((T,vocab_size))
			for right_answer in Y[senten_i]:
				loss += -np.log(output_prob[right_answer])  #<--- This (np.log) op is unstable!
		return loss 
	"""
	
	outputs = []

	for senten_i in xrange(len(Y)):
		_, output_prob = forward_propagation(X[senten_i])
		outputs.append(tf.squeeze(output_prob))
		# shape of output_prob = (sentence_length, vocab_size) 
	logits = tf.concat(outputs, 0) # shape=(number_of_words, 8000)

	train_label = []
	for labels_i in xrange(len(Y)):
		one_hot_label = tf.nn.embedding_lookup(np.eye(vocab_size), Y[labels_i])
		one_hot_label = tf.reshape(one_hot_label, [int(one_hot_label.shape[0]), vocab_size, 1])
		train_label.append(tf.squeeze(one_hot_label))    

	train_labels =  tf.concat(train_label, 0) 

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
							labels=tf.concat(train_labels, 0), logits=logits))   
	""" Shape! 
						In [311]: train_labels
						Out[311]: <tf.Tensor 'concat_26:0' shape=(192, 8000) dtype=float64>

						In [312]: logits
						Out[312]: <tf.Tensor 'concat_25:0' shape=(192, 8000) dtype=float32>
	"""
	#######################
	# Set up the optimizer!
	#######################
	global_step = tf.Variable(0)
	learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	gradients, v = zip(*optimizer.compute_gradients(loss))
	gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
	optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

	# Predictions.
	train_prediction = tf.nn.softmax(logits)



end = time.time()
print(end - start)

###########################################################################################
# Session starts!

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  i=0
  for step in range(num_steps):
    #X=X_batch[i]
    #Y=Y_batch[i]
    i+=1
    feed_dict = {'X':X, 'Y':Y}
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
 
