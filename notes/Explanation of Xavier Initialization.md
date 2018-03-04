
#### Xavier Initialization of Weights

- The main intuition is that our inputs $$x$$ into our network have a certain distribution and variance, and we want our outputs, for example after passing our inputs through a layer, to have the same variance. 
- Xavier intialization helps in training deep neural networks because it propagates signals deeper through the network, resulting in gradients being propagated backwards through the network, and not resulting in the network failing due to the vanishing or exploding gradient problem. 
- Intuitively, it keeps signals from growing too large by keeping the weights small enough, and it keeps signals from being too small by keeping the weights large enough, at initialization time. This is done by trying to make sure tha the variance between each successive layer stays roughly the same, so that there are very few, if any, unusually large or unusually small signals being propagated through the network. 

#### Derivation
- The general approach to Xavier intialization is to initialize your weights with mean $$0$$ and variance $$\frac{1}{n}$$< where $$n$$ is the shape of the input to the layer (i.e. if your input has 784 dimensions like MNIST, you'd use $$n = 784$$). 
- To see why this makes sense, we can derive it below: 
- Like we mentioned above, we want $$Var(Y) = Var(X)$$. 
- Say that $$Y$$ is a linear combination of our inputs: $$Y = \sum_i w_ix_i$$. 
- Then, $$Var(y) = Var(\sum_i w_ix_i)$$. We assume that each feature is independent of the other, so $$Var(y) = \sum_{i=1}^{N} Var(w_ix_i)$$. 
- Next, we compute $$Var(w_ix_i)$$. To do this, we use the identify that $$Var(a) = E[a^2] - E[a]^2$$. Therefore, we have

$$Var(w_ix_i) = E[w_i^2x_i^2] - E[w_ix_i]^2$$. 

We invoke an independence assumption again, namely that the weights and features are independent of each other, so that $$E[w_i^2x_i^2] =  E[w_i^2]E[x_i^2]$$. Therefore, we have 

$$Var(w_ix_i) = E[w_i^2]E[x_i^2] - (E[w_i]E[x_i])^2$$. 

Next, since we assume that our data and weights are centered around $$0$$, we have $$E[w_i] = E[x_i] = 0$$ and

$$Var(w_ix_i) = E[w_i^2]E[x_i^2]$$. 

But since $$Var(w_i) = E[w_i^2] - E[w_i]^2 = E[w_i^2] $$ (and same for $$x$$), we have 

$$Var(w_ix_i) = Var(w_i)Var(x_i)$$. 

Therefore, we have 

$$Var(y) = \sum_{i=1}^{N} Var(w_i)Var(x_i) = nVar(w_i)Var(x_i)$$. 

Since we want $$Var(y) = Var(x_i)$$, we simply initialize $$Var(w_i) = \frac{1}{b}$$.



```python
-
```
