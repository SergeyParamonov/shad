import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import BernoulliNB

def main():
  data = pd.read_csv("data.csv")
  avgs = data.mean()
  binary = data.copy()
  for avg,var in zip(avgs,["v1","v2","v3","v4"]):
    binary[var] = data[var] > avg
  binary['class'] = make_pure_label(binary['label'])

  apply_bayes("alearn","blearn","aexam","bexam",binary)
  apply_bayes("alearn","clearn","aexam","cexam",binary)
  apply_bayes("blearn","clearn","bexam","cexam",binary)

def make_pure_label(label):
  return label[0]

make_pure_label = np.vectorize(make_pure_label)


def split_by_labels(learn1, learn2, exam1, exam2, data):
  train = data[np.logical_or(data['label'] == learn1, data['label'] == learn2)]
  y_train = train['class']
  train.drop("label",1,inplace=True)
  train.drop("class",1,inplace=True)
  test   = data[np.logical_or(data['label'] == exam1, data['label'] == exam2)]
  y_test = test['class']
  test.drop("label",1,inplace=True)
  test.drop("class",1,inplace=True)
  return train,y_train,test,y_test


def apply_bayes(learn1, learn2, exam1, exam2, binary):
  class1 = learn1[0]
  class2 = learn2[0]
  print(class1, " vs ", class2)
  data = binary.copy()
  data.iscopy = False
  train, y_train, test, y_test = split_by_labels(learn1, learn2, exam1, exam2, binary)
  clf = BernoulliNB()
  clf.fit(train, y_train)

  predicted = clf.predict(test)
  print(confusion_matrix(predicted,y_test))
  print(accuracy_score(predicted,y_test)) 

if __name__ == "__main__":
  main()
