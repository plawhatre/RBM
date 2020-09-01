import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from colorama import init
from termcolor import *
import copy
init()

class SymmSNE:
	def __init__(self, **kwargs):
		'''
		Here, I have calculated sigma using an approximation of 
		shanon entropy for gaussian rather than a binary search.
		One can change this  since this is just a hyperparmeter.
		'''
		try:
			param = kwargs['perplexity']
			self.sigma_2 = 1/(2*np.pi) * np.exp(2*np.log2(param)-1)
			cprint(f'sigma**2 = {self.sigma_2}', 'blue')
		except:
			self.sigma_2 = kwargs['sigma']**2

		self.X = None
		self.Y = None
		self.N = None
		self.D = None
		self.rD = kwargs['dim']
		self.P = None
		self.Q = None
		self.normlization_P = None
		self.normlization_Q = None
		cprint('Model Initialized', 'red')

	def Pj_and_i(self, X, i, j, sigma_2):
		num = np.exp(
			-np.linalg.norm(X[i,:]-X[j,:])**2/(2*sigma_2))

		return num / self.normlization_P

	def Qj_and_i(self, Y, i, j):
		num = np.exp(
			-np.linalg.norm(Y[i,:]-Y[j,:])**2)
		
		return num / self.normlization_Q

	def forward(self):
		#joint probability normalization in higher dimension
		X = copy.deepcopy(self.X)
		den = 1e-4
		for k in range(self.N):
			den += np.sum(
				np.exp(-np.linalg.norm(X[k,:]-np.delete(X, k, axis=0),ord='fro', axis=(0,1))**2/(2*self.sigma_2)))
		self.normlization_P = den

		#joint probability normalization in higher dimension
		Y = copy.deepcopy(self.Y)
		den = 1e-4
		for k in range(self.N):
			den += np.sum(
				np.exp(-np.linalg.norm(Y[k,:]-np.delete(Y, k, axis=0),ord='fro', axis=(0,1))**2))
		self.normlization_Q = den

		for i in range(self.N):
			for j in range(self.N):
				self.P[i,j] = self.Pj_and_i(self.X, i, j, self.sigma_2)
				self.Q[i,j] = self.Qj_and_i(self.Y, i, j)

	def grad(self):
		np.fill_diagonal(self.P, 0.0)
		np.fill_diagonal(self.Q, 0.0)
		M = np.repeat(self.P - self.Q, repeats=self.rD, axis=1)
		Y_tilda =  np.repeat(self.Y.reshape(1,- 1), repeats=self.N, axis=0) - np.repeat(self.Y, repeats=self.N, axis=1)
		gradient = 4*np.sum(np.multiply(M, Y_tilda), axis=0).reshape(np.shape(self.Y))
		return gradient

	def gradient_update(self, lr=0.001):
		self.forward()
		self.Y -= lr*self.grad()

	def normalise(self, a):
		mean = np.mean(a, axis=0)
		std = np.std(a, axis=0)
		x = (a - mean) / std
		return x

	def fit_transform(self, X, epochs=5, lr=0.005):
		self.N, self.D = X.shape
		self.X = X
		self.Y = np.random.randn(self.N, self.rD)
		self.P = np.zeros(shape=(self.N, self.N), dtype=np.float32)
		self.Q = np.zeros(shape=(self.N, self.N), dtype=np.float32)
	
		cprint('Training on data........','blue')
		for i in range(epochs):
			self.gradient_update()
			cprint(f'Epoch: {i+1}', 'green')

		self.Y = self.normalise(self.Y)
		return self.Y
	
if __name__ =='__main__':
	Data, color = datasets.make_circles(n_samples=100, factor=0.1, noise=0)
	Model = SymmSNE(sigma=0.5, dim=2)
	Y = Model.fit_transform(Data, epochs=20, lr=0.01)
	plt.figure()
	plt.subplot(121)
	plt.scatter(Data[:,0], Data[:,1], c=color)
	plt.subplot(122)
	plt.scatter(Y[:,0], Y[:,1], c=color)
	plt.show()