import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from train import Train
from src.utils import Utils

"""
  Main method.
"""
if __name__ == '__main__':
  data = pd.read_csv('data/data.csv')

  utils = Utils()
  x, y = utils.balance_classes(data)

  x_train, x_test, y_train, y_test = utils.split_data(x,y)
  train = Train(x_train, y_train, x_test, y_test)

  acc = train.train_stack()
  print(acc)

  #y = pd.read_csv('data/test.csv')

  #print(corr.head())
  #sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
  #plt.show()