from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 100000)
from sklearn.svm import SVC
linsvm = SVC(C = 0.1, kernel='linear', verbose = True)
linsvm.fit(X,y)
preds = linsvm.predict(X)
from sklearn.metrics import accuracy_score
acc = accuracy_score(preds, y)
print acc
rbfsvm = SVC(C = 0.1, kernel = 'rbf', verbose = True)
rbfsvm.fit(X,y)
preds = rbfsvm.predict(X)
acc = accuracy_score(preds, y)
print acc
