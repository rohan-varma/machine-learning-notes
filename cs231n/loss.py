import numpy as np

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