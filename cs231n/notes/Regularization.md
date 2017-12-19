### Regularization Basics

- Helps control overfitting by penalizing parameters according to a specific loss function that is a function of the weights. 
- Original loss function: $$L = \frac{1}{N} \sum_{i=1}^{N} L_i$$ 
- New, regularized loss function: $$L = \frac{1}{N} \sum_{i=1}^{N} L_i  + \lambda R(W)$$ where R(W) is a function of the weights .
- Example: L2 regularization: 
  - Penalize the weights by their L2 norm. This discourages large weights. If we have 2 sets of weights that perform the same, regularizing with l2 norm would prefer the smaller weights. 
  - **Why are smaller weights preferable? **
    - Consider linear regression where you have 2 sets of features that differ by an extremely small amount $$\epsilon$$ . So we'd have $$y_1 = Wx_1 + K$$ and $$y_2 = W(x_ 1 \epsilon) + K$$  , so the 2 predictions would differ by $$W\epsilon$$. Since these 2 feature vectors differ by only a very small amount we'd want their predictions to be close together by a model that has the ability to generalize, but if W were very large then this would not be true.
  - L2 penalty makes classifier more likely to take into account all of the input features and place a small weight on each, instead of placing larger weights on a few specific inputs, which is achievable through L1 regularization
- L2 regularization is equivalent to putting a gaussian prior on the W you are tying to estimate. L1 imposes a Laplacian prior.
- L1 regularization leads to sparse weights, which is good when you want a sparse model that only activates for a few specific features. It can also thus make the classifier work more like a feature detector - when certain features are detected then the model activities, and it does not activate when other features are present (since those weights are set to 0)

