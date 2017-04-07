import numpy as np
import matplotlib.pyplot as plt

# data generation
import sys
sys.path.append('../data')
from make_data import make
N, D, K = 100, 2, 3 # number of pts per class, # of dims, # of labels
X, y = make(N, D, K)
# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

def ss(z):
	z-= np.max(z)
	exps = np.exp(z)
	return exps / np.sum(exps, axis = 1, keepdims = True)

def forward(X, W, b, W2, b2):
	hidden_layer = np.maximum(0, np.dot(X, W) + b)
	return ss(np.dot(hidden_layer, W2) + b2), hidden_layer


def backward(dscores, hidden_layer, W, b, W2, b2):
	dW2 = hidden_layer.T.dot(dscores)
	db2 = np.sum(dscores, axis = 0, keepdims = True)
	dh = dscores.dot(W2.T)
	dh[hidden_layer<=0]=0
	dW = X.T.dot(dh)
	db = np.sum(dh, axis = 0, keepdims = True)
	return dW2, db2, dW, db

# gradient descent loop
num_examples = X.shape[0]
for i in range(10000):
  probs, hidden_layer = forward(X, W, b, W2, b2)  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 1000 == 0:
    print ("iteration %d: loss %f" % (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  dW2, db2, dW, db = backward(dscores, hidden_layer, W, b, W2, b2)
  # add regularization gradient contribution
  dW2 += reg * W2
  dW += reg * W
  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  W2 += -step_size * dW2
  b2 += -step_size * db2


# evaluate training set accuracy
hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))
