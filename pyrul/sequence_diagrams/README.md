# pyrul framework

pyrul allows for :
- building a dataset
- building a model
- fitting the model
- evaluating the model

For this, we consider the K-Fold cross-validation split to build a train-validation datasets.


The benchmarking process for pyrul can be visualized with the following sequence diagrams.

## 1. Building the training and validation datasets
![Alt text](build_and_fit_model.png?raw=true)
The raw data is first loaded into the memory. Then, it is split using K-fold to get the training and validation datasets for each fold.
After the split, the training and validation data are transformed, once the basic local preprocessing algorithm is fitted to the training data.
The same is done for the global preprocessing process, after which we obtained the final train and validation datasets.
After the preprocessing, we compute an average local metrics set, that uses all fold metrics for the future test dataset preprocessing.
Typically for instance, the mean of feature means and variances are computed.

## 2. Building the test datasets
![Alt text](build_test_data.png?raw=true)
After the local and global preprocessing steps have been fitted to the training datasets, it is possible to load and preprocess the test dataset.
For full cover, we consider 2 datasets :
- the test dataset where each sample is used to predict the last RUL of the respective unit
- the full test dataset where every sample is available

## 3. Building and fitting the models
![Alt text](build_and_fit_model.png?raw=true)
Once the training and validation datasets are built, the models of each fold can be instanciated and fitted to their respective data.
The models can then be assessed on the desired dataset via the Evaluator class. Visualization utilities are also available.

## 5. Hyperparameter search
![Alt text](run_hp_search.png?raw=true)
The hyperparameter search uses Optuna studies, where for a given search space, Grid or Bayesian search is carried out to find the best set of hyperparameters for a given model. The search essentially consists in a loop where, for a hyperparameter candidate set, the data is built and the models are fitted and assessed on the validation dataset. The best hyperparameter set is the one for which the average validation metric was the lowest across all trials.

## 6. Utilities
![Alt text](optional_loading_and_saving.png?raw=true)
The DataLoader allows users to save and load many objects at any time, including :
- data metrics
- datasets
- models