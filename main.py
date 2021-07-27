import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.utils import Utils

"""
  Main method.
"""
if __name__ == '__main__':
  utils = Utils()

  data = pd.read_csv('data/training.csv')
  x, y = utils.balance_classes(data)
  x_train, x_test, y_train, y_test = utils.split_data(x,y)

  clf = RandomForestClassifier(max_depth=5, random_state=10)
  clf.fit(x_train, y_train)

  scores = cross_val_score(clf, x_test, y_test, cv=10)
  print(np.mean(scores))

  '''
  fig, ax = plt.subplots()
  sns.countplot('Label', data=x, ax=ax)
  ax.set_title("Fraude")
  ax.set_xlabel("Valor")
  ax.set_ylabel("Contagem")
  plt.show()
  '''

  #y = pd.read_csv('data/test.csv')

  #print(corr.head())
  #sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
  #plt.show()