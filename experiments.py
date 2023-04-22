import json
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from templates import FeatureExtractor, logger
from classical_models import Classical_Model
from tqdm import tqdm

def evaluate(y_test, y_pred, notes):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

  result = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'tp': tp,
    'tn': tn,
    'fp': fp,
    'fn': fn
  }

  result.update(notes)
  return result


def save_results(results_dict, dest_file, *args, **kwargs):
  header=True
  for key, value in kwargs.items():
    logger.info("{} is {}".format(key,value))
    if key == 'header':
      header=value
  
  results_df = pd.DataFrame(results_dict)
  if dest_file.is_file():
    df = pd.read_csv(dest_file)
    results_df = pd.concat([df, results_df])
    logger.info('Appending to the existing .csv file.')

  results_df.to_csv(dest_file,  index=False, header=header) 

class TestManager:
  def __init__(self, data_manager, config, output_file_name=''):
    self.data_manager = data_manager
    self.results = []
    self.all_params = []
    self.config = config
    self.feature_extractors_dict = {}
    self.models_dict = {}
    self.output_file_name = output_file_name

  def __evaluations(self, y_test, y_pred, model, feature_extractor):
    notes = {
      'feature_method': model.feature_method,
      'model': model.model_name
    }

    notes.update(self.data_manager.notes)
    notes.update(feature_extractor.notes)
    notes.update(model.notes)
    notes.update({'dataset': self.config['dataset']['file']})
    self.results.append(evaluate(y_test, y_pred, notes))
    

  def __features_models_cartesian_tests(self, feature_methods, models):
    total_feature_extractors = len(feature_methods)
    total_models = len(models)
    idx_fe = 1
    
    for feature_method in tqdm(feature_methods, desc='Feature extractors'):
      logger.info(f'running {idx_fe} of {total_feature_extractors} feature extractors.')
      idx_fe += 1
      feature_extractor = FeatureExtractor(feature_method)
      feature_extractor.extract_features(
        self.data_manager.x_train, self.data_manager.x_test)
      self.feature_extractors_dict.update({feature_extractor.method: feature_extractor})
      idx_model = 1
      for model_name in tqdm(models, desc='Models'):
        logger.info(f'running {idx_model} of {total_models} models.')
        idx_model += 1
        model = Classical_Model(model_name)
        model.feature_method = feature_extractor.method
        model.fit(
          feature_extractor.features['train'], self.data_manager.y_train)
        model.vectorizer = feature_extractor.vectorizer 
        
        if model.model_name not in self.models_dict.keys():
          self.models_dict[model.model_name] = {}
        self.models_dict[model.model_name].update({feature_extractor.method: model})
        
        # Save the trained model
        # 
        
        if int(self.config['settings']['save_models']) != 0:
          # save model only if it was not saved before:
          if (model.model_name, feature_extractor.method) not in [(name,fe) for name, fe, _ in self.all_params]:
            timestamp = int(time.time())
            file_name = (Path(self.config['models']['dir']) 
                        / f"{model.model_name}_{feature_extractor.method}_{timestamp}.pkl")
            model.save_model(file_name)

        y_pred = model.predict(feature_extractor.features['test'])
        self.__evaluations(self.data_manager.y_test, y_pred, model, feature_extractor)
        params = model.model.get_params()
        #Problem: XGboost as a parameter of OneVsRest, cannot be serialized with json.dump
        #dirty fix: just save the name, not the object
        if 'estimator' in params:
          params['estimator'] = type(params['estimator']).__name__
        
        # save parameters only if it was not saved before:
        if (model.model_name, feature_extractor.method) not in [(name,fe) for name, fe, _ in self.all_params]:
          self.__append_model_params_file(model_name=model_name, fe=feature_extractor.method, model_params=params)
          
  def __append_model_params_file(self, model_name, fe, model_params):
    self.all_params.append((model_name, fe, model_params))
  
  def __save_all_params_file(self, dir):
    if self.output_file_name == '':
      currentDateAndTime = datetime.now()
      currentTime = currentDateAndTime.strftime("%y%m%d_%H%M%S")
      file_name = Path(dir) / f'results_{currentTime}.csv'
    else:
      file_name = Path(dir) / Path(self.output_file_name).name
    
    self.output_file_name = file_name
    param_file_name = file_name.parent / file_name.with_suffix('.json').name.replace('results', 'params')
    
    # Save the parameters to a file
    with open(param_file_name, 'w') as f:
      json.dump(self.all_params, f)
    logger.info(f"Params saved to {param_file_name}")

  def __save_results(self, dir):
    if self.output_file_name == '':
      currentDateAndTime = datetime.now()
      currentTime = currentDateAndTime.strftime("%y%m%d_%H%M%S")
      file_name = Path(dir) / f'results_{currentTime}.csv'
    else:
      file_name = Path(dir) / Path(self.output_file_name).name
    
    self.output_file_name = file_name
    save_results(self.results, dest_file=self.output_file_name, header=True)
    logger.info(f"Results saved to {self.output_file_name}")

  def run_tests(self, feature_methods, classic_models):
    self.__features_models_cartesian_tests(feature_methods, classic_models)
    #self.__run_ensemble_tests(ensemble_models)
    #self.__adaptive('tf-idf_ngram', 'xgboost', threshold=0.5)
    #adaptive_model = self.__adaptive('tf-idf_ngram', 'xgboost', threshold=0.3)
    #file_name = self.__save_pred_prob(adaptive_model,dir=Path(self.config['results']['dir']))
    # TODO: save pred_prob and open it in display results ipython file.
    # TODO: plot ROC for 0.0001 values as well(may be logarithmic in x axis?)
    # TODO: Make a function to calculate the estimated speed vs Recall.
    #self.__plot_roc3(file_name)

    self.__save_results(Path(self.config['results']['dir']))
    self.__save_all_params_file(Path(self.config['results']['dir']))
    self.results = []