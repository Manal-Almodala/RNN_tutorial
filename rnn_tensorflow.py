
import tensorflow as tf 


graph

as default graph


hidden_dim = 100
vocab_size = 8000


graph = tf.Graph()
with graph.as_default():
	#Parameters
	U = tf.Variable(tf.truncated_normal([num_nodes, vocab_size]), -0.1, 0.1)
	W = tf.Variable(tf.truncated_normal([num_nodes, num_nodes]), -0.1, 0.1)
	V = tf.Variable(tf.truncated_normal([vocab_size, num_nodes]), -0.1, 0.1)


	def forward_propagation(X):
		# input  : list of index of words
		# output : hidden state & output (probability of next words) 
		T = len(X)
		S = tf.zeros((T+1,hidden_dim))
		O = tf.zeros((T,vocab_size))
		
		for t in xrange(T):
			S[t] = tf.tanh( U[:,X[t]] + tf.matmul(W,S[t-1]) )
			O[t] = tf.nn.softmax(tf.matmul(S,V))
		return S, O

	def predict(X):
		s, o = forward_propagation(X)
		return 


	
