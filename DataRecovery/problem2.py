import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from problem1 import split_by_labels, make_pure_label
from itertools import chain


def multiply(x,w):
  t = np.array(list(chain([1.],x)))
  return np.dot(t,w)

def scale_y(y, class1):
  new_y = np.array(list(map(lambda x: 1 if x == class1 else -1, y)))
  return new_y


def apply_perceptron(learn1, learn2, exam1, exam2, data):
  class1 = learn1[0]
  class2 = learn2[0]
  print(class1," vs ", class2)
  d = 4 # number of dimensions
  X_train, y_train, X_test, y_test = split_by_labels(learn1, learn2, exam1, exam2, data)
  X_train = X_train.as_matrix()
  X_test  = X_test.as_matrix()
  y_train = scale_y(y_train,class1)
  y_test  = scale_y(y_test,class1)
  step = 0
  tolerance = 25
  previous_count = 10**5
  w = np.ones(d+1)
  while True:
#   print(previous_count)
#   print("w",w)
    step += 1
    grad = np.zeros(d+1)
    if tolerance < 0:
      break
    count = 0
    for x,y in zip(X_train, y_train):
      if y*multiply(x,w) <= 0:
          grad -= y*np.array(list(chain([1],x))) 
          count += 1
    if previous_count == count or count < np.ceil(len(y_train)*0.05):
      tolerance -= 1
    previous_count = count
    normalized = normalize(grad)[0]
#   print("grad",normalized)
    w -= normalized
  print(count)
  print(w)
  test_count = 0
  for x,y in zip(X_test,y_test):
    if multiply(x,w)*y < 0:
      test_count += 1
  print(1-test_count/len(y_test))


def main():
  data = pd.read_csv("data.csv")
  data['class'] = make_pure_label(data['label'])
  apply_perceptron("alearn","blearn","aexam","bexam",data)
  apply_perceptron("alearn","clearn","aexam","cexam",data)
  apply_perceptron("blearn","clearn","bexam","cexam",data)

if __name__ == "__main__":
  main()
