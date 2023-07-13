<a target="_blank" href="https://colab.research.google.com/github/gdrlab/dpu-sqli-detection/blob/main/main.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# SQLi Detection on DPU
This repository provides the codes and the dataset to regenerate the results given in the paper: K. Tasdemir et al., "Detecting SQL Injection Through Classical Machine Learning on DPU," _2023 IEEE 36th International System-on-Chip Conference (SOCC)_, Santa Clara, CA, USA, 2023, pp.(TBA) 
![single_nlp_f1_vs_time](https://github.com/gdrlab/dpu-sqli-detection/assets/6195512/080c4166-1da8-4d0e-ab08-57a557e784ab)


## Requirements  
Note: The code doesn't check if DPU exists. It can run on machines with or without DPU.
- (optional, easier) Jupyter Notebook, or Google Colab account if you just want to see how the code runs, or view the results.
- (option B) If you want to run on local machines with DPU on them, use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge). Mamba package search is significantly faster than Anaconda (or [Anaconda](https://www.anaconda.com/products/distribution) environment)
## Installation and Setup
### (option B) For local setup 
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
- Modify config.ini file
- run 'main.ipynb' on Colab

### (Optional) Running only the classical ML based methods
- Activate sqli-env.yml environment in the Miniforge prompt and run the test:
    - ``` python run_classical_MLs.py -o <output file path>```


## Demonstrating the experimental results

- Use ``` main.ipynb ```.

## Folder contents
- Main folders: (datasets, trained_models, results)
  - datasets: SQLi csv file with two columns:'payload' and 'label'.
  - config.ini: choose the models to be tested and other options. Note: Ensemble models need all classical MLs to be run before.
  - results: .csv and .json file outputs of the experimental tests.

## Troubleshot

- 
## Release notes
- Release (v1.0.0)
  - Only main.ipynb is now enough for tests and demonstrations.

## TODO
- Save the trained models. Add a method to load and run the saved models without training.

## Acknowledgment
- (TODO: Add the project code)
- Kaggle SQLi [dataset](https://www.kaggle.com/datasets/syedsaqlainhussain/sql-injection-dataset)

 
