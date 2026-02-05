![Alt text](pyrul_bandroll.png?raw=true)

# pyrul : Remaining Useful Life Benchmark

pyrul is an open local Python library for remaining useful life (RUL) estimation benchmarking on CMAPSS FD001.

## I. Benchmark

### I.1. Models
- Reproduced RUL prediction models are from the following works:
    - CMAPSS FD001 Dataset:
        - Direct Remaining Useful Life Estimation Based on Support Vector Regression (Racha Khelif, 2017)
        - Long Short-Term Memory Network for Remaining Useful Life Estimation (Shuai Zheng, 2017)
        - Deep separable convolutional network for remaining useful life prediction of machinery (Biao Wang, 2019)
        - Remaining useful life estimation with multiple local similarities (Jianhua Lyu, 2020)
        - Trajectory Similarity Matching and Remaining Useful Life Prediction Based on Dynamic Time Warping (Lin Huan, 2022)
        - Variational encoding approach for interpretable assessment of remaining useful life estimation (Nahuel Costa, 2022)
        - Dual Aspect Self-Attention based on Transformer for Remaining Useful Life Prediction (Zhizheng Zhang, 2022)
          
    - FEMTO/PRONOSTIA Dataset:
        - Remaining useful life (rul) prediction of rolling element bearing using random forest and gradient boosting technique (Sangram Patil, 2018)
        - Remaining useful life estimation of bearing via temporal convolutional networks enhanced by a gated convolutional unit (Yujie Qin, 2024)

### I.2. Results

Obtained results on CMAPSS FD001 test data across 10 seeds are given in the following table.

| Model | RMSE (x10) | MAE (x10) | Scoring (x10²) | IT (s) |
| :---: | :--------: | :-------: | :------------: | :----: |
| LR | 2.564 ± 0.001 | 2.098 ± 0.001 | 27.27 ± 0.06 | .002 ± .001 |
| SVR | 2.656 ± 0.029 | 1.947 ± 0.021 | 163.07 ± 20.90 | .032 ± .005 |
| MLS | 3.240 ± 0.051 | 2.431 ± 0.036 | 47.05 ± 4.62 | 4726.4 ± 53.7 |
| TSBPDTW | 2.321 ± 0.008 | 1.823 ± 0.009 | 8.64 ± 0.12 | 107.9 ± 0.5 |
| DSCNN | 1.478 ± 0.027 | 1.127 ± 0.030 | 4.05 ± 0.32 | 16.9 ± 0.4 |
| DeepLSTM | 1.580 ± 0.045 | 1.157 ± 0.046 | 4.89 ± 0.07 | 5.1 ± 0.2 |
| RULRVE | 1.484 ± 0.019 | 1.133 ± 0.024 | **3.43 ± 0.16** | 5.7 ± 0.3 |
| DAST | **1.453 ± 0.013** | **1.068 ± 0.011** | 3.51 ± 0.14 | 30.5 ± 2.4 |
| | | | | |
| GBT | 3.145 ± 0.016 | 2.318 ± 0.022 | 240.94 ± 22.77 | .005 ± .001 |
| GCUTCN | 2.700 ± 0.090 | 1.904 ± 0.078 | 126.05 ± 27.74 | 21.3 ± 2.3 |

## II. Running Models

### II.1. Installation
Please note that pyrul was tested on the following computer settings :
- Windows 11 64x
- Intel(R) Core(TM) Ultra 7 165U
- GPU Nvidia A30 (CUDA 12.2 / nvidia-cudnn-cu12 9.1.0.70)

<br>Important dependencies</br>
Regardless of the package manager, the most important dependencies to install for pyrul to work are listed below :
- tensorflow 2.16.1
- optuna 4.2.1
- torch 2.6.0
- scikit-learn 1.6.1
- scipy 1.15.1
- pandas 2.2.3
- numpy 1.26.4
- matplotlib 3.10.0
- seaborn 0.13.2
- dtw-python 1.5.3
- tslearn 0.6.3

Below, we provide instructions for quickly installing pyrul dependencies with pip or uv, <u>under the aforementionned computer settings</u>. To utilize the GPU under tensorflow, please follow the recommended instructions for GPU support at https://www.tensorflow.org/install/pip?hl=en.

<br>Installing with pip</br>

Download the repository and create a conda environnement with python 3.11.11, then, when inside the main repository directory, run <code>pip install -r requirements.txt</code>.
Both tensorflow and pytorch are required, and may take up some time to download.

<br>Installing with uv</br>

Download the repository and run <code>uv init python 3.11.11</code>, then <code>uv python pin 3.11.11</code>, then <code>uv sync</code>.

### II.2. Working with pyrul

Once installed, the 3 essential notebooks to create, train and evaluate models are located inside the Model folder, that includes:
- Model.ipynb : A starter to understand the main pyrul objects
- Model HP search.ipynb : The hyperparameter search algorithm
- Model Evaluation.ipynb : The evaluation of trained models.

## II.2. About pyrul

The unified framework of the benchmark is a bagging approach on top of a k-fold cross-validation scheme.
The operations behind the pyrul code are visually represented by several sequence diagrams located in pyrul/sequence_diagrams, and further details on the framework can be read in our related article.
