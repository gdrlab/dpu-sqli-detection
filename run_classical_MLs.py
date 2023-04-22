import configparser
from templates import DataManager, logger
from experiments import TestManager
import sys, getopt
from tqdm import tqdm


def main(argv):
  opts, args = getopt.getopt(argv,"ho:",["ofile="])
  outputfile = ''
  for opt, arg in opts:
    if opt == '-h':
      print ('main.py -o <output_file_name>')
      sys.exit()
    elif opt in ("-o", "--ofile"):
      outputfile = arg

  config = configparser.ConfigParser()
  config.read('config.ini')
  data_manager = DataManager(config)
  test_manager = TestManager(data_manager=data_manager,config=config, output_file_name=outputfile)

  feature_methods = [method.strip() for method in config.get('feature_methods', 'methods').split(',')]
  classic_models = [model.strip() for model in config.get('models', 'classic_models').split(',')]
  #ensemble_models = [model.strip() for model in config.get('models', 'ensemble_models').split(',')]

  seed_idx = 0
  total_seeds = len(config.get('data_manager', 'seed').split(','))
  with tqdm(total=total_seeds, desc='Seeds') as pbar:
    while (data_manager.split_data(seed_idx=seed_idx)): #while there are more seeds  
      logger.info(f'Running the tests for seed: #{seed_idx} ({data_manager.seed}) of {total_seeds} seeds')
      test_manager.run_tests(feature_methods, classic_models)
      seed_idx += 1
      pbar.update(1)

  return test_manager.output_file_name
  


if __name__ == "__main__":
  main(sys.argv[1:])
