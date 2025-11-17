# base.py
# The RULModel class template.

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import optuna

from pyrul.utils.file import clear_tmp_plots, SamplerStateSaveCallback, SaveBestDataOnly
from pyrul.utils.dataset_info import get_default_folds

class RULModel(ABC):
    """
        Remaining Useful Life prediction model framework. Contains data preprocessing steps and the core of the model.
    """

    def __init__(self, 
                 dataset_name,
                 dataset_ori,
                 folds=None, 
                 seed=None,
                 is_ibl=False,
                 is_pred_masked=False,
                 is_deep_lstm=False,
                 is_rul_normalized=False):
        """
            Parameters
            ----------
            dataset_name : str
                The name of the dataset to consider.

            dataset_ori : str
                The original name of the dataset the model was designed with.

            folds : int, default=None
                The number of folds for k-fold split. If None, uses the default values depending on dataset_name.

            is_ibl : bool, default=False
                Whether the model is an instance based learning one. These models only predict the last RUL value of a given trajectory (seq2one).

            is_deep_lstm : bool, default=False
                Whether the model is a sequence-to-sequence model with masked labels (masked seq2seq).

            is_rul_normalized : bool, default=False
                Whether the RUL labels are normalized.

            seed : int
                The seed, default=None.
        """
        # General attributes
        self._dataset_name = dataset_name
        self._dataset_ori = dataset_ori
        self._FOLDS = get_default_folds(dataset_name) if folds is None else folds
            
        self._SEED = seed

        # Data attributes
        self._df = None
        self._train_datasets = None
        self._val_datasets = None
        self._test_dataset = None
        self._test_full_dataset = None

        # Models
        self._models = None

        # Model specific attributes
        self._is_ibl = is_ibl
        self._is_deep_lstm = is_deep_lstm
        self._is_rul_normalized = is_rul_normalized

    ###############################################################
    ##################### DATA PREPROCESSING ######################
    ###############################################################

    @abstractmethod
    def data_process_local_fit_and_transform(self, 
                                             df: pd.DataFrame, 
                                             k: int,
                                             )->list:
        """
            Fits the preprocessing pipeline to the training dataset of the current fold.

            Parameters
            ----------
            df : pd.DataFrame
                The training dataset.

            k : int
                The index of the fold associated to the dataset df.

            Returns
            -------
            train_datasets : list
                The transformed training datasets.
        """
        pass
        
    @abstractmethod
    def data_process_local_transform(self, 
                                     df: pd.DataFrame,
                                     k: int,
                                     data_type: str
                                     )->list:
        """
            Transforms the given dataset using the training data' metrics.

            Parameters
            ----------
            df : pd.DataFrame
                The training dataset.

            k : int
                The index of the fold associated to the training data' metrics.
                If k == -1, uses the already computed average training data' metrics.

            data_type : str
                The data type. Is either "train", "val", "test" or "test_full".

            Returns
            -------
            dataset : list
                The transformed dataset.
        """
        pass

    @abstractmethod
    def data_process_global_fit_and_transform(self, 
                                              train_datasets: list, 
                                              val_datasets: list
                                              )->dict:
        """
            Transforms the training datasets using all training folds and the validation data.
            Training data is changed into a dictionary of lists.

            Parameters
            ----------
            train_datasets : list
                The training datasets.

            val_datasets : list
                The validation datasets.

            Returns
            -------
            train_datasets : dict
                The final transformed training datasets.
        """
        pass
        
    @abstractmethod
    def data_process_global_transform(self, 
                                      datasets: list, 
                                      data_type: str,
                                      )->dict:
        """
            Transforms the given datasets using all training folds.
            Data is changed into a dictionary of lists.

            Parameters
            ----------
            datasets : dict or list
                The datasets to transform.

            data_type : str
                The data type. Is either "train", "val", "test" or "test_full".

            Returns
            ----------
            datasets : dict
                The final modified datasets.
        """
        pass
    
    @abstractmethod
    def data_process_save_metrics(self, 
                                  path: str,
                                  )->None:
        """
            Saves the averaged local training data metrics using all folds.

            Parameters
            ----------
            path : str
                The path to save the datasets' metrics.
        """
        pass

    @abstractmethod
    def data_process_load_metrics(self,
                                  path: str,
                                  )->None:
        """
            Loads the averaged local training data metrics and the global data metrics.

            Parameters
            ----------
            path : str
                The path to load the datasets's metrics.
        """
        pass

    @abstractmethod
    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        pass

    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    @abstractmethod
    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        pass

    @abstractmethod
    def fit_model(self)->None:
        """
            Fits each model to their respective training fold.
        """
        pass

    def predict(self, 
                x: np.array, 
                k: int, 
                do_mean: bool=True,
                is_test: bool=False,
                )->np.array:
        """
            Returns the prediction associated to the input x, for the k-th model.
            If k==-1, a list of all model predictions is returned, or the average of it
            depending on do_mean.

            Parameters
            ----------
            x : np.array
                The data input.
    
            k : int
                The index of the model to use.
    
            do_mean : bool, default=True
                Whether to get the average prediction in the case of getting predictions from all models.

            is_test : bool, default=False
                Whether the data is test or not.

            Returns
            ----------
            preds : np.array
                The model's RUL predictions.
        """
        if k<-1 or k>self._FOLDS:
            raise ValueError("The given fold index is invalid. Possible values are from -1 to",self._FOLDS,". Received k=",k)
        else:
            if k!=-1:
                return self._models[k].predict(x)
            else:
                preds = [self._models[k].predict(x) for k in range(self._FOLDS)]
                if do_mean:
                    return np.mean(preds, axis=0)
                else:
                    return preds

    @abstractmethod
    def get_hp_trial(self, 
                     dataloader, 
                     evaluator,
                     data_load_path: str,
                     **kwargs
                     )->callable:
        """
            Returns the function to call to run the trial of the hyperparameter search.

            Parameters
            ----------            
            dataloader : DataLoader
                The dataloader object.

            evaluator : Evaluator
                The evaluator object to compute the validation metric.

            data_load_path : str
                The path to load the data.

            Returns
            ----------   
            hp_trial : callable
                The function to call for the study.
        """
        pass

    def run_hp_search(self,
                      sampler: optuna.samplers.BaseSampler,
                      save_db_path: str, 
                      study_name: str,
                      dataloader,
                      evaluator,
                      data_load_path: str,
                      best_data_save_path: str,
                      best_data_metrics_save_path: str,
                      best_model_save_path: str,
                      best_trial_idx: int,
                      n_trials: int,
                      **kwargs: dict
                      )->optuna.study.Study:
        """
            Runs a hyperparameter search over n_trials trials using the given sampler.
            The search is saved in a database.
            Calling this function while a study database already exists at the saving location
            will resume the study. The sampler's inner state is also saved after each trial.
            best_trial_idx should be manually updated when resuming a study. 

            Parameters
            ----------
            sampler : optuna.samplers.BaseSampler
                The sampler to use.

            save_db_path : str
                The path to save the database.

            study_name : str
                The name of the study.

            data_load_path : str
                The path to load the data.

            best_data_save_path : str
                The path to save the best model's datasets.

            best_data_metrics_save_path : str
                The path to save the best model's data metrics.

            best_model_save_path : str
                The path to save the best model.

            best_trial_idx : int
                The best trial index so far. Put -1 for the first time running the function.

            n_trials : int
                The number of trials to run.

            **kwargs
            Optional keyword arguments to pass to model.get_hp_trial().

            Returns
            ----------
            study : optuna.study.Study
                The finished hyperparameter search study.
        """
        study = optuna.create_study(direction='minimize', 
                                    sampler=sampler,
                                    storage=save_db_path,
                                    study_name=study_name,
                                    load_if_exists=True)
        
        study.optimize(self.get_hp_trial(dataloader=dataloader, 
                                         evaluator=evaluator, 
                                         data_load_path=data_load_path,
                                         **kwargs),
                       n_trials=n_trials, 
                       callbacks=[SamplerStateSaveCallback(), 
                                  SaveBestDataOnly(dl=dataloader,
                                                    model=self,
                                                    data_path=best_data_save_path, 
                                                    data_metrics_path=best_data_metrics_save_path, 
                                                    model_path=best_model_save_path,
                                                    best_trial_idx=best_trial_idx)
                                 ]
                      )
        return study

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################

    def get_dict_dataset(self, 
                         dataset: list, 
                         is_folded: bool
                         )->dict:
        """
            Converts the dataset into a dictionnary form.
            Final training and validation datasets are dict containing "fold_data", "fold_labels" and "fold_lengths" which are lists of arrays of samples for each fold.
            Final test and test_full datasets are dict containing "data", "labels" and "lengths" which are arrays of samples.
            This function should be called before any model training.
    
            Parameters
            ----------
            dataset : list
                The dataset list containing the features "X", the labels "y" and the original unit lengths "lengths".

            is_folded : bool
                Whether the dataset comes from K-fold.

            Returns
            ----------
            dataset : dict
                The new dataset.
        """
        if type(dataset)!=dict:
            if is_folded:
                new_dataset = dict({"fold_data":[], "fold_labels":[], "fold_lengths":[]})
                for k in range(self._FOLDS):
                    X, y, lengths = dataset[k]
                    new_dataset["fold_data"].append(X)
                    new_dataset["fold_labels"].append(y)
                    new_dataset["fold_lengths"].append(lengths)
            else:
                X, y, lengths = dataset
                new_dataset = dict({"data":X, "labels":y, "lengths":lengths})
            return new_dataset
            
        else:
            return dataset
    
    @abstractmethod
    def get_slices(self, 
                   idx: int, 
                   lengths: list,
                   data_type: str
                   )->tuple:
        """
            Returns 2 indices to select the idx-th unit.
            Beware that due to data preprocess, the indices of time series do not correspond to the original units'.

            Parameters
            ----------  
            idx : int
                The index of the fold's original time series. 

            lengths : list
                Lengths of each unit.

            data_type : str
                Is either "train", "val", "test" or "test_full".

            Returns
            ----------
            slice_inf, slice_sup : tuple
                The indices to select the desired unit.
        """
        pass

    @abstractmethod
    def get_cycles(self, 
                   length: int,
                   is_test: bool
                   )->np.array:
        """
            Returns the cycles associated to the trajectory's unit given its length.

            Parameters
            ----------
            length : int
                The length of the trajectory of the related unit.

            is_test : bool
                Whether the trajectory is in the full test dataset or not.

            Returns
            ----------
            cycles : np.array
                The time steps associated to the unit.
        """
        pass
        
    @abstractmethod
    def save_hp_values(self,
                       path: str
                       )->None:
        """
            Saves the current hyperparameter values in a csv file.

            Parameters
            ----------
            path : str
                The path to save the hyperparameter values.
        """
        pass

    @abstractmethod
    def save_config(self,
                    path: str
                    )->None:
        """
            Saves the attribute values of the model in a csv file, and indicates which one has been tuned.

            Parameters
            ----------
            path : str
                The path to save the hyperparameter values.
        """
        pass