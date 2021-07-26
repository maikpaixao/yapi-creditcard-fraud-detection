import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
  training_data = pd.read_csv('data/training.csv')
  test_data = pd.read_csv('data/test.csv')

  x_train = training_data.iloc[:, :-4]
  y_train = training_data['Label']

  x_test = test_data.iloc[:, :-4]
  y_test = test_data['Label']

  clf = RandomForestClassifier(max_depth=2, random_state=0)
  clf.fit(x_train, y_train)

  scores = cross_val_score(clf, x_test, y_test, cv=5)
  print(scores)

  #y = pd.read_csv('data/test.csv')

  #print(corr.head())
  #sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
  #plt.show()