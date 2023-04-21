from templates import Model
import xgboost as xgb
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Classical_Model(Model):
  def __init__(self, model_name, *args, **kwargs):
    super().__init__( model_name, *args, **kwargs)
    self.results = []

  def _create_model(self, *args, **kwargs):
    if self.model_name == 'xgboost':
      self.model = xgb.XGBClassifier(*args, **kwargs)
    elif self.model_name == 'svm':
      self.model = svm.SVC(*args, **kwargs)
    elif self.model_name == 'naive_bayes':
      self.model = MultinomialNB(*args, **kwargs)
    elif self.model_name == 'KNeighborsClassifier':
      self.model = KNeighborsClassifier(3, *args, **kwargs)
    elif self.model_name == 'SVC-GC':
      self.model = SVC(gamma=2, C=1, *args, **kwargs)
    elif self.model_name == 'GaussianProcessClassifier':
      self.model = GaussianProcessClassifier(1.0 * RBF(1.0), *args, **kwargs)
    elif self.model_name == 'DecisionTreeClassifier':
      self.model = DecisionTreeClassifier(max_depth=5, *args, **kwargs)
    elif self.model_name == 'RandomForestClassifier':
      self.model = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, *args, **kwargs)
    elif self.model_name == 'MLPClassifier':
      self.model = MLPClassifier(alpha=1, max_iter=1000, *args, **kwargs)
    elif self.model_name == 'AdaBoostClassifier':
      self.model = AdaBoostClassifier( *args, **kwargs)
    elif self.model_name == 'GaussianNB':
      self.model = GaussianNB( *args, **kwargs)
    elif self.model_name == 'QuadraticDiscriminantAnalysis':
      self.model = QuadraticDiscriminantAnalysis( *args, **kwargs)
    else:
      raise ValueError(f"Unknown model: {self.model_name}")
    return 