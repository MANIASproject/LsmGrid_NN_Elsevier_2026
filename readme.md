# LsmGrid

This project corresponds to the revised version of LsmGrid (*LsmGrid: Compact Liquid State Machines from Optimized Spatiotemporal Grids*).

LsmGrid is a methodology for building ultra-compact reservoirs for Liquid State Machines.

This is a research project. The code provided here is intended to ensure the reproducibility of the scientific paper, not for commercial or industrial use. It is not a Python library, but rather the code containing the experiments and analyses carried out.

## Project structure

- `lsmgrid_experiment.ipynb` contains the experiments performed in the paper. This is the main file of the project.
- `spikes_study.ipynb` and `table_stats.ipynb` are used to generate analyses and tables reported in the paper.
- `./analysis` contains scripts for processing the experimental results and extracting the information used in the paper.
- `./grids` contains the logic related to the grids, their tree representation, and their conversion into a spiking neural network.
- `./optimizer` contains the logic related to the HC-GP, Parser+GA, and Parser+(A)CSA optimizers.
- `./results` contains the files with the experimental results.
- `./tunning` contains the hyperparameter tuning of the optimizers using irace.
- `./utils` contains several functions required to visualize grids, evaluate the objective function, and manage the results.
- `./data` contains everything related to the preprocessing and loading of the datasets.

## Requirements

The required dependencies are listed in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

## Dataset setup

Four datasets were used in this project. N-TIDIGITS and N-MNIST were downloaded through the Python library `tonic`, so no files need to be downloaded manually.

- The FSDD dataset can be downloaded from https://www.kaggle.com/datasets/joserzapata/free-spoken-digit-dataset-fsdd?resource=download
- The MIT-BIH dataset can be downloaded from https://www.kaggle.com/datasets/mondejar/mitbih-database

For FSDD, the `.wav` files must be placed in the `./data/recordings` folder. For MIT-BIH, the `.csv` files and `annotations.txt` must be placed in the `./data/mit-bih` folder.

The datasets must be preprocessed before running the main file. Preprocessing should generate a file named `./data/DATASETNAME_preprocessed.pickle`, which is the file read by the main notebook.

## Contact

For any comments, please write to ai.romo.moena@upm.es