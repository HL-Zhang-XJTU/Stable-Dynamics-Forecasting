# Stable-Dynamics-Forecasting

This github repository contains the codebase accompanying the paper "Long-Horizon Stable UAV Dynamics Forecasting with Neural Lyapunov Functions and  Spatial-Temporal Linear Transformer".

## System Requirements
This repository was developed and has been tested  with Python 3.6, PyTorch 1.10, and CUDA 11.3.

## Code Overview
This repository is structured as follows:

```
 ├──data                                    - Location for storing datasets.
 │    ├──stanford_helicopter_data           - Stanford autonomous helicopter flight Dataset.
 ├──result                                  - Location for storing the training results.
 │    ├──SGDM                               - Results for the trained SGDM model.
 │    ├──STLT                               - Results for the trained STLT model.
 ├──script                                  - Main experiment scripts.
``` 
## Scripts
- **load_data.py**: loading the flight data.
- **model_SGDM.py**: the stability-guaranteed dynamics modeling (SGDM) method.
- **model_STLT.py**: the spatial-temporal linear Transformer (STLT) neural network.
- **model_NLF.py**: the network for the neural Lyapunov function (NLF).
- **train_sgd.py**: training the SGDM.
- **train_stlt.py**: training the STLT.
- **test.py**: evaluating the trained model.
- **utils.py**: containing the implementation of some common functions.

## Dataset
Our algorithm was validated on a public helicopter dataset and a private quadrotor flight dataset.
Since the private dataset was collected jointly by us and another company, they initially agreed to let us use it in our paper for algorithm validation. However, due to commercial confidentiality concerns, they did not consent to fully disclosing this dataset. We are currently in negotiations with them regarding data disclosure and have not yet obtained their approval. 
Once they agree to make the dataset public, we will immediately add the download link to this code repository.

Here, we are only providing the public dataset for those interested in reproducing our experiments.
The public dataset is the Stanford Helicopter Data.
You can download the dataset from the link  
http://heli.stanford.edu/dataset/.
After downloading, unzip it and place it into folder "data".


## Training
Please run `train_SGDM.py` or `train_STLT.py` in the `script` folder. After training the model, some simple statistics and plots are generated which show the model performance fitting to the training and validating data. 

**Note:** Modify the parameter settings at the beginning of the file, as these hyperparameters can be used to configure different network architectures and training methods. The best checkpoint will be made available after the paper is accepted.


## Evaluation 
Please run `test.py` in the `script` folder. 

**Note:** Modify the parameter settings at the beginning of the file  to make them consistent with those used in the training process.
