# SQLi Detection on DPU
This repository provides the codes and the dataset to regenerate the results given in the paper: K. Tasdemir et al., "Detecting SQL Injection Through Classical Machine Learning on DPU," _2023 IEEE 36th International System-on-Chip Conference (SOCC)_, Santa Clara, CA, USA, 2023, pp.(TBA) 
![single_nlp_f1_vs_time](https://github.com/gdrlab/dpu-sqli-detection/assets/6195512/080c4166-1da8-4d0e-ab08-57a557e784ab)


## Requirements  

- (recommended) Google Colab account, or any other Jupyter Notebook.
- (optional) If you want to run on local machines, use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge). Mamba package search is significantly faster than Anaconda (or [Anaconda](https://www.anaconda.com/products/distribution) environment)
## Installation and Setup
### For Google Colab setup
- Upload all GitHub files into your Google Drive (e.g. '/content/drive/MyDrive/Akademik/Research and Projects/Kasim/AI Security Intelligence/Codes/20230311_sqli_colab')
- Update the hardcoded paths and run 'main.ipynb'
### (optional) For local setup 
You can skip this part if you use Google Colab. 

The Classical ML methods can run on virtual Python environment on Win, Mac or Linux.

- Download and install the [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) environment.
- Open Miniforge Prompt and change current directory to the project folder. 
- Run the following command in the folder, where **sqli-env.yml** file resides. This will create a new Python environment with the required packages:
    -  ``` mamba env create -f sqli-env.yml ```
- Activate the environment
    - ``` mamba activate sqli-env.yml ```


## Running
### Running all
- Modify config.ini file and run 'main.ipynb'

### (Optional) Running only the classical ML based methods
- Activate sqli-env.yml environment in the Miniforge prompt and run the test:
    - ``` python run_classical_MLs.py -o <output file path>```


## Demonstrating the experimental results

- Modify ``` utils\Demonstrate_test_results.ipynb ``` to point the result file and run it.

## Folder contents
- Main folders: (datasets, utils, trained_models, results)
  - datasets: SQLi csv file with two columns:'payload' and 'label'.
  - config.ini: choose the models to be tested and other options. Note: Ensemble models need all classical MLs to be run before.
  - utils   
    - Demonstrate test results : produce all visuals and tables used in the paper.

## Troubleshot

- 
## Release notes
- Release (v0.5.0)
  - Adaptive method is implemented. XGboost is being trained with 5000x positive weight, 0.05 threshold in inference.
  - The proposed method is added to the test results (Tables and Figures)
  - Supports multiple random seeds in config.ini file.
- Release (v0.4.0)
  - OOP is used to modularize the code.
  - Most of the settings can be set from config.ini file.
  - Classic ML and Transformers can be run with a single file (notebook)
  - "results demonstration" file is updated
  - results are saved into CSV file, not pkl.
  - "saving models" does NOT work.
- Release (0.3.0) Latex table generation
  - The results are saved to a pkl file in the main folder.
  - the results pickle files can be read and visualized using utils\Demonstrate_test_results.ipynb. This also generates the Latex tables used in the paper.
  - the visualized results have color scheme. The fonts sizes, etc. are ready for the paper.
- Release (v0.1.0)
    - Results are saved to a Pandas dataframe. It is saved to a pickle file.
    - Results can be visualized using Utils/Data visualize . jpy notebook.
    - The original code required datasets with 'delimiter=three tabs'. This is no longer supported by Pandas data frame. So, It has been changed to support single tab delimited dataset. If you need to use the old code on the old datasets, you can use Release v0.0.12
	- utils/clean-kaggle-sqli-dataset-and-split.ipynb file is created for cleaning Kaggle SQLi dataset and splitting it into train-test files.
	- utils/convert-old-dataset-to-new-single-tab.ipynb file is created for converting old three tabs delimited dataset to single tab.
	- utils/dataset-train-test-splitter.ipynb file is created for splitting the given dataset to train and test parts.
	- support for running without building the package is added (nlp_hybrid.py)

- Release v0.0.12

    - This is the original , the first code from Rafi. It works with Python 3 (the very first one was Python 2)

## TODO
- Save the trained models. Add a method to load and run the saved models without training.

## Acknowledgment
- (TODO: Add the project code)
- Kaggle SQLi [dataset](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)

 
