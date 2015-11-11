import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from problem1 import split_by_labels, make_pure_label
from itertools import chain

def kernel(a,x,y):
  d = x - y
  power = -1*a*norm(d)
  return np.exp(power)

def classify(point,a,X,Y,class1,class2):
  potential = 0
  for x,y in zip(X,Y):
    if y == class1: 
      r = 1
    else:
      r = -1
    potential += r*kernel(a,point,x)
  if potential >= 0:
    return class1
  else:
    return class2

def apply_potentials(learn1, learn2, exam1, exam2, data, a):
  class1 = learn1[0]
  class2 = learn2[0]
  print(class1," vs ", class2)
  X_train, y_train, X_test, y_test = split_by_labels(learn1, learn2, exam1, exam2, data)
  X_train = X_train.as_matrix()
  X_test  = X_test.as_matrix()
  predicted = []
  for x in X_test:
    predicted.append(classify(x,a,X_train,y_train,class1,class2))
  predicted = np.array(predicted)
  print(sum(predicted == y_test)/len(y_test))
  
    
    


data = pd.read_csv("data.csv")
data['class'] = make_pure_label(data['label'])
apply_potentials("alearn","blearn","aexam","bexam",data,0.01)
apply_potentials("alearn","clearn","aexam","cexam",data,0.01)
apply_potentials("blearn","clearn","bexam","cexam",data,0.01)

apply_potentials("alearn","blearn","aexam","bexam",data,1)
apply_potentials("alearn","clearn","aexam","cexam",data,1)
apply_potentials("blearn","clearn","bexam","cexam",data,1)
