
#### Dying ReLu Problem

- ReLu units have several advantages over sigmoid/tanh nonlinearities, the main one being that ReLu is nonsaturating and doesn't have the vanishing gradient/exploding gradient problem that these units can have. ReLu's have also been shown empirically to train networks much faster (i.e. the AlexNet paper said that with relu the network trained "several times faster"
- A downside of ReLus, however is the dying ReLu problem. 
- So given a neural network, the network (parametrized by the weights) represent some probability distribution over the weights.
- i.e. At a simple level, the softmax classifier represents $p(y_i | x_i) = softmax_{y_i}(W^Tx)$. 
- Therefore, the input into each layer and each unit can also be thought of as some probability distribution. Let's assume that the distribution of a certain unit is a Gaussian with small variance centered around 0.1. This means that most values are small and positive, so the ReLu gate is open, and the gradient flows through it during backprop. 
- However, suppose that there is some input that for whatever reason causes a very large gradient to be passed to the unit. Then assuming the input into this unit $x > 0$, this large gradient will be multiplied by 1 and passed through
- This may cause a large change in the distribution that approximates the inputs to R, so that it's now a low variance Gaussian with mean $-0.1$. 
- This means that many of the inputs passed to this relu will be $ x < 0$, which means that no gradient will be passed through backwards, meaning that this ReLu unit won't be able to update its params. 


```python

```
