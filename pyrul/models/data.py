# data.py
# To load and process the dataset for the given model.
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pyrul.models.base import RULModel
from tqdm import tqdm
import random

class DataLoader():
    """
        Class to handle data loading, saving and preprocessing associated to a RULModel object.
    """

    def __init__(self, 
                 dataset: str
                ):
        """
            Parameters
            ----------
            dataset : str
                The name of the dataset to consider.
        """
        self._dataset = dataset

        if self._dataset not in ["cmapss_fd001", "femto"]:
            raise ValueError("Dataset '",self._dataset,"'name not found.")

    ###############################################################
    ##################### DATA LOADING ############################
    ###############################################################
    
    def get_raw_data(self, 
                     data_path: str
                    )->pd.DataFrame:
        """
            Loads the complete run-to-failure dataset features and labels.
            Raw data has units ordered.

            Parameters
            ----------
            data_path : str
                The path to load the data.
        
            Returns
            -------
            df : pd.DataFrame
                The dataframe.
        """
        if self._dataset=="cmapss_fd001":
            df = pd.read_csv(data_path, sep=" ", index_col=0)
            
        elif self._dataset=="femto":
            df = pd.read_csv(data_path, sep=",")

        return df.astype(float)
        
    def load_data_metrics(self,
                          model: RULModel,
                          path: str,
                           )->None:
        """
            Loads the data metrics of the model.
            
            Parameters
            ----------
            model : RULModel
                The prediction model.
            
            path : str
                The path to load the datasets' metrics.
        """
        model.data_process_load_metrics(path=path)

    ###############################################################
    ##################### DATA PREPROCESSING ######################
    ###############################################################

    def build_train_val_data(self,
                             model: RULModel,
                             data_path: str
                            )->None:
        """
            Loads the data, applies a k-fold cross-validation split and preprocesses each fold to create the training and validation datasets. Finally, a global fitting and transform process on applied.

            Parameters
            ----------
            model : RULModel
                The prediction model.
                
            data_path : str
                The path to the data.
        """
        # Loading, splitting and preprocessing the data
        model._df = self.get_raw_data(data_path)
        self.process_train_val_data(model)

    def build_test_data(self, 
                        model: RULModel,
                        data_path: str
                       )->None:
        """
            Loads the test data and preprocesses it to create the test datasets.

            Parameters
            ----------
            model : RULModel
                The prediction model.
                
            data_path : str
                The path to the data.
        """
        # Loading and preprocessing the data
        model._df_test = self.get_raw_data(data_path)
        self.process_test_data(model)

    def apply_k_fold_CV(self, 
                        model: RULModel,
                        nb_folds: int,
                        seed: int=None
                       )->tuple:
        """
            Applies a k-fold cross validation split on the units.
            Returns two sets of training and validation datasets inside two dictionnaries
            each containing the "fold_data", "fold_labels" and "fold_lengths" arrays.

            Parameters
            ----------
            nb_folds : int
                The number of folds to create.
    
            seed : int, default=None
                The seed to fix.

            Returns
            -------
            train_datasets : list
                The training datasets object.

            val_datasets : list
                The validation datasets object.
        """       
        # K-Fold Cross-validation (on the units)
        unit_list = model._df['unit'].unique()
        kf = KFold(n_splits=nb_folds, shuffle=True, random_state=seed)
        
        train_datasets = []
        val_datasets = []
        
        for i, (train_index, val_index) in enumerate(tqdm(kf.split(unit_list), total=kf.get_n_splits(), desc="k-fold")):

            df_train = model._df[~model._df["unit"].isin(val_index+1)].copy() # preserves unit order
            df_val = model._df[model._df["unit"].isin(val_index+1)].copy() # preserves unit order

            # Shuffling units
            ids = df_train["unit"].unique()
            random.shuffle(ids)
            df_train = df_train.set_index("unit").loc[ids].reset_index()

            ids = df_val["unit"].unique()
            random.shuffle(ids)
            df_val = df_val.set_index("unit").loc[ids].reset_index()
            
            train_dataset = model.data_process_local_fit_and_transform(df_train, k=i)
            val_dataset = model.data_process_local_transform(df_val, k=i, data_type="val")
            
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)

        return train_datasets, val_datasets

    def process_train_val_data(self, 
                               model: RULModel
                              )->None:
        """
            Creates the training and validation datasets of the model.

            Parameters
            ----------
            model : RULModel
                The prediction model.
        """
                    
        # K-fold CV split where we apply local transformation (preprocessing steps for each fold)
        model._train_datasets, model._val_datasets = self.apply_k_fold_CV(model=model,
                                                                          nb_folds=model._FOLDS, 
                                                                          seed=model._SEED)
        model.compute_avg_local_metrics()
        
        # Global transformation
        model._train_datasets = model.data_process_global_fit_and_transform(model._train_datasets, model._val_datasets)
        model._val_datasets = model.data_process_global_transform(model._val_datasets, data_type="val")
        
    def process_test_data(self, 
                          model: RULModel
                         )->None:
        """
            Creates the processed test data (normal and full version) of the model.

            Parameters
            ----------
            model : RULModel
                The prediction model.
        """
        test_dataset = dict({"data":[], "labels":[]})

        test_dataset = model.data_process_local_transform(df=model._df_test.copy(), 
                                                           k=-1,
                                                           data_type="test")

        test_full_dataset = model.data_process_local_transform(df=model._df_test.copy(), 
                                                                k=-1,
                                                                data_type="test_full")

        model._test_dataset = model.data_process_global_transform(test_dataset, data_type="test")
        model._test_full_dataset = model.data_process_global_transform(test_full_dataset, data_type="test_full")

    ###############################################################
    ##################### DATA SAVING #############################
    ###############################################################

    def save_datasets(self,
                      model: RULModel,
                      data: list,
                      path: str,
                     )->None:
        """
            Saves the desired datasets of the model.

            Parameters
            ----------
            model : RULModel
                The prediction model.

            data : list of str
                The list of dataset names to save.

            path : str
                The path to save the datasets.        
        """
        for d in data:
            if d == "train":
                np.save(path+"_train.npy", model._train_datasets)
            elif d == "val":
                np.save(path+"_val.npy", model._val_datasets)
            elif d == "test":
                np.save(path+"_test.npy", model._test_dataset)
            elif d == "test_full":
                np.save(path+"_test_full.npy", model._test_full_dataset)
            else:
                raise ValueError("Data '{d}' not found.".format(d=d))

    def load_datasets(self,
                      model: RULModel,
                      data: list,
                      path: str,
                     )->None:
        """
            Loads the desired datasets of the model.

            Parameters
            ----------
            model : RULModel
                The prediction model.

            data : list of str
                The list of dataset names to load.

            path : str
                The path to load the datasets.        
        """
        for d in data:
            if d == "train":
                model._train_datasets = np.load(path+"_train.npy", allow_pickle=True).item()
            elif d == "val":
                model._val_datasets = np.load(path+"_val.npy", allow_pickle=True).item()
            elif d == "test":
                model._test_dataset = np.load(path+"_test.npy", allow_pickle=True).item()
            elif d == "test_full":
                model._test_full_dataset = np.load(path+"_test_full.npy", allow_pickle=True).item()
            else:
                raise ValueError("Data '{d}' not found.".format(d=d))

    def save_data_metrics(self,
                          model: RULModel,
                          path: str,
                           )->None:
        """
            Saves the data metrics of the model.

            Parameters
            ----------
            model : RULModel
                The prediction model.

            path : str
                The path to save the datasets' metrics.       
        """
        model.data_process_save_metrics(path=path)