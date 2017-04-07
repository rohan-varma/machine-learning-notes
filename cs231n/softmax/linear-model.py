import numpy as np
import matplotlib.pyplot as plt

# data generation
import sys
sys.path.append('../data')
from make_data import make
N, D, K = 100, 2, 3 # number of pts per class, # of classes, number of dimensions
X, y = make(N, D, K)


def stable_softmax(z):
	z-=np.max(z)
	exps = np.exp(z)
	return exps / np.sum(exps, axis = 1, keepdims = True)

def forward(X, W, b):
	scores = np.dot(X, W) + b
	normed = stable_softmax(scores)
	return normed

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

def train(X, W, b, num_examples):
	for i in range(200):
	  probs = forward(X, W, b)
	  # compute the loss: average cross-entropy loss and regularization
	  corect_logprobs = -np.log(probs[range(num_examples),y])
	  data_loss = np.sum(corect_logprobs)/num_examples
	  reg_loss = 0.5*reg*np.sum(W**2)
	  loss = data_loss + reg_loss
	  if i % 10 == 0:
	    print("iteration %d: loss %f" % (i, loss))
	  
	  # compute the gradient on scores
	  dscores = probs
	  dscores[range(num_examples),y] -= 1
	  dscores /= num_examples
	  
	  # backpropate the gradient to the parameters (W,b)
	  dW = np.dot(X.T, dscores)
	  db = np.sum(dscores, axis=0, keepdims=True)
	  
	  dW += reg*W # regularization gradient
	  
	  # perform a parameter update
	  W += -step_size * dW
	  b += -step_size * db
	return W, b

	
W, b = train(X, W, b, num_examples = X.shape[0])


scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

