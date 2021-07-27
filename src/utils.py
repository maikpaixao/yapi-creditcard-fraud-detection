
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

class Utils:
  """
  Class Utils
  """
  def __init__(self):
    self.data = []
  
  def balance_classes(self, data):
    """
    Method to balance dataset.
    :param data: Data required to be balanced.
    """
    x = data.iloc[:, :-4]
    y = data['Label']
    under_sampler = RandomUnderSampler(random_state=10)
    return under_sampler.fit_resample(x, y)

  def split_data(self, x,y):
    """
    Method to split dataset.
    :param x: Dependent variables
    :param y: Idependent variable
    """
    return train_test_split(x, y, test_size=0.2, random_state=10)
