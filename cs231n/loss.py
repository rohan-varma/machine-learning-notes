import numpy as np
# Multiclass SVM loss implementation
def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class
  correct_class_score = scores[y]
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0
  for j in xrange(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  print(scores)
  print(scores[y])
  exit()
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i

def one_hot(y, num_classes):
  # one-hot encode y so that y[:][i] gives a vector of size num_classes where each element is one-hot
  y_one_hot = np.zeros((num_classes, len(y)))
  for idx, val in enumerate(y):
    # element idx has label val
    y_one_hot[val][idx] = 1.0
  return y_one_hot

def L(X, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
  delta = 1.0
  n_examples = X.shape[1]
  scores = W.dot(X) # 10 rows * 50k columns, each scores[:][i] will give the predictions for example i
  # one-hot encode the labels y
  y_one_hot = one_hot(y, max(y) + 1)
  assert scores.shape == y_one_hot.shape
  margins = scores - y_one_hot + 1
  margins[margins < 0] = 0
  # we sum across 50k examples and count the correct label 50k times, so our loss is 50k more than it should be
  return np.sum(margins) - n_examples


def make_softmax_graph(x_probs):
  # precondition: x is an np.array of increasing probs between 0 and 1
  # the label will be 0, 1, ... len(x_probs) - 1
  x_probs = np.arange(start = 0.01, stop = 1.0, step = 0.02)
  labels = np.array([0 for _ in x_probs])
  softmax_one = lambda x, labels: -(labels.dot(np.log(x_probs)))
  softmax_two = lambda x, labels: -(labels.dot(np.log(x_probs)) + (1-labels).dot(np.log(1-x_probs)))
  first_way, second_way = [], []
  print(len(x_probs))
  for idx, _ in enumerate(x_probs):
    # assign the correct label
    if idx == 0:
      labels[idx] = 1
    else:
      labels[idx-1], labels[idx] = 0, 1 #deassign the old label
    first_way.append(softmax_one(x_probs, labels))
    second_way.append(softmax_two(x_probs, labels))
  # verify the losses are at least monotic dec
  # and softmax 2 loss is higher than softmax 1 loss? 
  first_less_than_second = [val[0] < val[1] for val in zip(first_way, second_way)]
  monotonic_dec = [idx == 0 or first_way[idx] < first_way[idx-1] and second_way[idx] < second_way[idx-1] for idx in range(min(len(first_way), len(second_way)))]
  assert all(first_less_than_second)
  assert all(monotonic_dec)
  import matplotlib.pyplot as plt
  line_first, = plt.plot(x_probs, first_way)
  line_second, = plt.plot(x_probs, second_way)
  print('abs diff in loss for first way {}, for second way {}'.format(first_way[0]-first_way[-1], second_way[0]-second_way[-1]))
  line_first.set_label('y_j * log(h(x_i))')
  line_second.set_label('y_j * log(h(x_i)) + (1-y_j) * log(1 - h(x_i))')
  plt.legend()
  plt.xlabel('probability of correct class')
  plt.ylabel('loss')
  plt.show()
  return first_way, second_way




if __name__ == '__main__':
  # have 3 classes
  x_probs = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
  labels = np.array([0 for _ in x_probs])
  labels[0] = 1
  softmax_one = lambda x, labels: -(labels.dot(np.log(x_probs)))
  assert softmax_one(x_probs, labels) == -np.log(0.2), "output {} but got {}".format(softmax_one(x_probs, labels), np.log(0.2))
  print(softmax_one(x_probs, labels))
  softmax_two = lambda x, labels: -(labels.dot(np.log(x_probs)) + (1-labels).dot(np.log(1-x_probs)))
  print(-(np.log(0.2) + np.sum(np.log(1-np.array(x_probs.tolist()[1:])))))
  print(softmax_two(x_probs, labels))
  make_softmax_graph(x_probs)
  exit()
  y = [2, 3, 1, 0, 9]
  num_classes = 10
  print(one_hot(y, 10))
  # let X = 10 * 5 (10 features, 5 examples)
  # let W = 10 * 10 
  # let y be a 10 * 5 one hot encoded
  X = np.random.rand(10, 5)
  W = np.random.rand(10, 10)
  y = [0, 8, 6, 3, 9]
  print(L(X, y, W))



