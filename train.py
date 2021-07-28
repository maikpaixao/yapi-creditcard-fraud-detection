import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

class Train:
  """
  Class Train
  """
  def __init__(self, x_train, y_train, x_test, y_test):
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test

  def get_scores(self, model):
    """
    Method to cross validate model.
    :param model: Model determined for cross_validation.
    """
    return np.mean(cross_val_score(model, self.x_test, self.y_test, cv=10))

  def train_stack(self):
    """
    Method to train stacked estimator.
    """
    estimators = [('rf', RandomForestClassifier(n_estimators=10, max_depth=3, random_state=10)),
                   ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=10)))]

    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    model.fit(self.x_train, self.y_train)
    
    return self.get_scores(model)
