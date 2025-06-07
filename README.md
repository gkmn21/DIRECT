# DIRECT
Source code for paper 'DIRECT: Deep Reinforcement Learning for Tourist Route Recommendation'

# DIRECT: Deep Reinforcement Learning for Tourist Route Recommendation

This folder contains the source code for the paper **DIRECT: Deep Reinforcement Learning for Tourist Route Recommendation**.

## Folders
- `SSLP`, `USLP` - contain codes for the models SSLP and USLP respectively
- `Datasets` - contains TD1, TD2 and ID1 datasets.
- `Ground Truth Generation` - contains groundtruth generation scripts for creating TD1, TD2 and ID1 datasets.


## Dependencies
The `requirements.txt` has been provided for creating a venv environment with all dependencies. To install the requirements run:
```
pip install -r requirements.txt
```

Dependencies include:
- Python 3.12.4
- PyTorch 2.7.0
- `stable-baselines3`
- `gymnasium`

## Dataset Generation
The folder `dataset_generation` contains scripts for dataset generation.

- Run script `prepare_data.py` to fetch data for a region. The generated dataset files will be present in `data` folder.

## DIRECT model
- Run script `run_model.py` to train the model.
- Run script `inference.py` to run inference of the test set.
- Finally, run script `eval.py` to obtain the evaluation metrics.

### Baselines
- The scripts for naive baselines SP and GP is present in the folder `Baselines`.
- 2. Source code provided by the publications was used for baselines RB&C and D-RL


