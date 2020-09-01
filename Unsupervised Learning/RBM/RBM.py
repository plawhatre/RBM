import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from colorama import init
from termcolor import *
init()

class RBM(tf.keras.layers.Layer):
	def __init__(self, D, M, f=lambda x: x):
		super(RBM, self).__init__()
		self.D = D
		self.M = M
		self.W = tf.Variable(tf.random.normal(shape=[D, M])* 2/ np.sqrt(M))
		self.b = tf.Variable(tf.zeros(shape=[D], dtype=tf.float32))
		self.c = tf.Variable(tf.zeros(shape=[M], dtype=tf.float32))
		self.Ph_given_v = None
		self.Pv_given_h = None
		self.v = None
		self.h = None
		self.v_CD1 = None
		self.cost = None

	def free_energy(self, v):
		t1 = -tf.matmul(v, tf.reshape(self.b, [-1,1])) 
		t2 = - tf.math.reduce_sum(tf.nn.softplus(self.c + tf.matmul(v,self.W)))
		F_v = t1 + t2
		return F_v

	def forward_input(self, v):
		self.Ph_given_v = tf.nn.sigmoid(tf.matmul(v, self.W) + self.c)
		r = tf.random.uniform(shape=tf.shape(self.Ph_given_v))
		h = tf.dtypes.cast((r < self.Ph_given_v), tf.float32)
		return h 

	def forward_logits(self, h):
		self.Pv_given_h =  tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.b)
		r = tf.random.uniform(shape=tf.shape(self.Pv_given_h))
		v = tf.dtypes.cast((r < self.Pv_given_h), tf.float32)
		return v

	def cost_free_energy(self):
		f1 = self.free_energy(self.v)
		f2 = self.free_energy(self.v_CD1)
		f = tf.math.reduce_mean(f1 - f2)
		return f

	def gradient_update(self, X, optimizer):
		with tf.GradientTape() as t:
			#contrastive divergence
			self.v = X
			self.h = self.forward_input(self.v)
			self.v_CD1 = self.forward_logits(self.h)
			#loss
			Loss = self.cost_free_energy()
			grads = t.gradient(Loss, [self.b, self.W, self.c])
		optimizer.apply_gradients(zip(grads, [self.b, self.W, self.c]))
		return Loss

	def fit(self, X, epochs=5, batch_size=500, lr=1e-3):
		N = X.shape[0]
		n_batches = N // batch_size
		cprint('Train model........','green')
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		cost_lst = []
		for i in range(epochs):
			np.random.shuffle(X)
			for j in range(n_batches):
				X_batch = X[(j*batch_size):((j+1)*batch_size)]
				Loss = self.gradient_update(X_batch, optimizer)
				cost_lst.append(Loss/batch_size)
				if j % 10 ==0:
					cprint(f'Epoch: {i+1}, Batch: {j}, Loss: {Loss}','green')
		return cost_lst

	def transform(self, X):
		self.forward_input(X)
		return self.Ph_given_v

if __name__ == '__main__':
	(x_train, y_train),  (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255, x_test / 255
	N_train, H, W = np.shape(x_train)
	N_test, H, W = np.shape(x_test)
	x_train = x_train.reshape(N_train, H*W) 
	x_test = x_test.reshape(N_test, H*W) 
	D = H*W
	x_train = x_train.astype(np.float32)
	x_test = x_test.astype(np.float32)
	X = np.concatenate((x_train, x_test), axis=0)
	Y = np.concatenate((y_train, y_test))
	M = 2
	model = RBM(D, M)
	cost_lst = model.fit(X, epochs=20)
	#lower dimensional projection
	X_hat = model.transform(X)
	plt.figure
	plt.subplot(121)
	plt.plot(cost_lst)
	plt.title('Loss Curve')
	plt.subplot(122)
	plt.scatter(X_hat[:,0], X_hat[:,1], c=Y)
	plt.title('Projected')
	plt.show()