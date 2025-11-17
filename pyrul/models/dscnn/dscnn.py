# dscnn.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from .dscnn_core import *
from pyrul.models.base import RULModel
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars


class DSCNN_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Deep separable convolutional network for remaining useful life prediction of machinery" (Biao Wang, 2019).
    """

    def __init__(self, 
                 DATASET_NAME: str = "cmapss_fd001",
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = None,
                 R_EARLY: float = 130,
                 WINDOW: int = 30,
                 BATCH_SIZE: int = 128,
                 EPOCHS: int = 100,
                 LEARNING_RATE: float = 1e-3,
                 SEED: int = None
                 )->None:
        """
            DATASET_NAME : str
                The name of the dataset to use.
            
            FOLDS : int
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depending on DATASET_NAME.

            R_EARLY : float, default=None
                The RUL threshold to apply to the train and validation labels. If None, no threshold is applied.

            WINDOW : int
                The size of the segments as input to the prediction model.

            BATCH_SIZE : int, default=128
                The size of each batch for the data.

            EPOCHS : int, default=100
                The number of epochs to train the model.

            LEARNING_RATE : float, default=1e-3
                The learning rate for ADAM to train the model.

            SEED : int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME, 
                         dataset_ori="cmapss_fd001",
                         folds=FOLDS,
                         seed=SEED)

        self._R_EARLY = R_EARLY
        self._WINDOW = WINDOW
        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._BATCH_SIZE = BATCH_SIZE
        self._EPOCHS = EPOCHS
        self._LEARNING_RATE = LEARNING_RATE

        # Data preprocess metrics
        self._df_train_mins = [None]*self._FOLDS
        self._df_train_maxs = [None]*self._FOLDS
        self._df_train_min = None
        self._df_train_max = None

    ###############################################################
    ##################### DATA PREPROCESSING ######################
    ###############################################################
    
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
        if k<=-1 or k>=self._FOLDS:
            raise ValueError("Invalid fold index. Allowed values range from 0 to "+str(self._FOLDS-1)+".")
            
        # Min-max norm
        self._df_train_mins[k] = df[self._VARS_TO_KEEP].min()
        self._df_train_maxs[k] = df[self._VARS_TO_KEEP].max()

        return self.data_process_local_transform(df=df, k=k, data_type="train")

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
        if k<-1 or k>=self._FOLDS:
            raise ValueError("Invalid fold index. Allowed values range from -1 to "+str(self._FOLDS-1)+".")
        if data_type not in ["train","val","test","test_full"]:
            raise ValueError("Data type '"+data_type+"' not found.")
            
        # Step 1. Piecewise Linear RUL labels
        apply_thres = data_type in ["train","val"]
        if apply_thres and self._R_EARLY is not None:
            df["RUL"] = np.minimum(self._R_EARLY, df["RUL"])

        # UNUSED # Scaling RUL for femto
        # if self._dataset_name == "femto":
        #     df_gb_rul = df.groupby("unit", sort=False)["RUL"]                
        #     df["RUL"] = df_gb_rul.transform(lambda x: x/(1+x.iloc[0])) # we divide by T, knowing RUL(t0)=T-1
    
        # Step 2. Feature selection and scaling (except for "unit" and "RUL")
        df_info = df[["unit","RUL"]].copy()
        df = df[self._VARS_TO_KEEP]

        if k!=-1:
            if (self._df_train_mins[k] is not None) and (self._df_train_maxs[k] is not None):
                df_train_min = self._df_train_mins[k]
                df_train_max = self._df_train_maxs[k]
            else:
                raise Exception("Data metrics are not built yet. Fit before.")
        else:
            if (self._df_train_min is not None) and (self._df_train_max is not None):
                df_train_min = self._df_train_min
                df_train_max = self._df_train_max
            else:
                raise Exception("Data metrics are not built yet. Load or fit before.")

        # Min-max norm
        df = 2*(df - df_train_min) / (df_train_max - df_train_min) - 1
            
        df[["unit","RUL"]] = df_info
    
        # Step 3. Individual pre-processing
        keep_last = data_type == "test"
        X, y, lengths = self.get_windows_and_labels(df, keep_last=keep_last)
    
        return [X, y, lengths]

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
        return self.data_process_global_transform(train_datasets, data_type="train")

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
        is_folded = data_type in ["train","val"]
        datasets = self.get_dict_dataset(datasets, is_folded=is_folded)
        return datasets
        
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
        self._df_train_min.to_csv(path+"_df_train_min.csv")
        self._df_train_max.to_csv(path+"_df_train_max.csv")

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
        self._df_train_min = pd.read_csv(path+"_df_train_min.csv", sep=",", header=0, index_col=0)['0']
        self._df_train_max = pd.read_csv(path+"_df_train_max.csv", sep=",", header=0, index_col=0)['0']
        
    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        self._df_train_min = pd.concat(self._df_train_mins, axis=1).mean(axis=1)
        self._df_train_max = pd.concat(self._df_train_maxs, axis=1).mean(axis=1)

    def get_windows_and_labels(self, 
                               ds: pd.DataFrame,
                               keep_last: bool=False,
                               )->tuple:
        """
            Returns the array of samples, labels and original length, such that each sample is a multi-dimensional window of the initial time series.
            Overlapping sliding window (step=1) is used to obtain the windows.
            Only takes the last possible window for units if keep_last=True.
    
            Parameters
            ----------
            ds : pd.DataFrame
                The dataset. Must have "unit" and "RUL" as columns at least.

            keep_last : bool, default=False
                Whether to keep the last sample of each unit or not. Should be 'True' for test data.
                
            Returns
            ----------
            X : np.array
                The feature values after window sliding.
            
            y : np.array
                The last RUL values of each window.
            
            lengths : np.array
                The lengths of the original unit.
        """
        # Time-window creation (assuming every lifetime is >= WINDOW)
        X, y, lengths = [], [], []
        unit_list = ds['unit'].unique()
        
        for unit in ds['unit'].unique():
            x = ds[ds['unit']==unit].copy()
    
            # Overlapping window slicing
            if keep_last:
                windows = [x[self._VARS_TO_KEEP].to_numpy()[-self._WINDOW:]]
                labels = x['RUL'].iloc[-1:]
                
            else:
                windows = [x[self._VARS_TO_KEEP].to_numpy()[k:k+self._WINDOW] for k in range(len(x)-self._WINDOW+1)]
                labels = x['RUL'].iloc[self._WINDOW-1:] # related labels (1 label = window's last cycle's RUL value)
    
            # Adding samples, labels and lengths
            X += windows
            y = np.concatenate([y, labels]) # labels related to each window
            lengths.append(len(x)) # lengths of original time serie
        
        # Final formatting
        X = np.array(X)
        y = np.array(y)
        
        X = tf.expand_dims(X, -2) # final shape : (batch, window, 1, channels) (specific for convolution usage)
        y = tf.expand_dims(y,-1) # final shape : (batch, 1)
        
        return X, y, lengths

    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        # Building the models
        self._models = [SepConvNet() for k in range(self._FOLDS)]
        optimizers = [tf.keras.optimizers.Adam(learning_rate=self._LEARNING_RATE) for k in range(self._FOLDS)]
        
        for k in range(self._FOLDS):
            if self._train_datasets is not None:
                _ = self._models[k](self._train_datasets["fold_data"][k][:2])
            elif self._test_dataset is not None:
                _ = self._models[k](self._test_dataset["data"][:2])
            else:
                raise AttributeError("Model has no defined train or test datasets to use.")

            self._models[k].compile(optimizer=optimizers[k], loss=tf.keras.losses.MeanSquaredError)

    def save_models(self, 
                    path: str
                    )->None:
        """
            Saves each model's state.

            Parameters
            ----------
            path : str
                The path to save the models.
        """
        if self._models is None:
            raise Exception("No models found. Call model.build_model() before.")
            
        for k in range(self._FOLDS):
            self._models[k].save_weights(path+'_model_'+str(k)+'.weights.h5')

    def load_models(self, 
                    path: str
                    )->None:
        """
            Loads each model's state.

            Parameters
            ----------
            path : str
                The path to save the models.
        """
        if self._models is None:
            raise Exception("No models found. Call model.build_model() before.")
            
        for k in range(self._FOLDS):
            load_path = path+"_model_"+str(k)+".weights.h5"
            self._models[k].load_weights(load_path)

    def fit_model(self, 
                  verbose: bool=True
                  )->None:
        """
            Fits each model to their respective training fold.

            Parameters
            ----------
            verbose : bool, default=True
                Whether to display information on the fitting process.

            Returns
            -------
            histories : list
                The list of every model fitting.
        """
        if self._models is None:
            raise Exception("No models found. Call model.build_model() before.")
            
        self._hist = []
        for k in tqdm(range(self._FOLDS), desc="fitting models"):
            X_train, y_train = self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k]
            X_val, y_val = self._val_datasets["fold_data"][k], self._val_datasets["fold_labels"][k]
        
            hist = self._models[k].fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 batch_size=self._BATCH_SIZE,
                                 epochs=self._EPOCHS,
                                 shuffle=True,
                                 verbose=verbose)
            self._hist.append(hist)

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

        def hp_trial(trial):
        
            self._LEARNING_RATE = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        
            set_seed(self._SEED)

            dataloader.build_train_val_data(self, data_path=data_load_path)
            
            self.build_model()
            self.fit_model(verbose=False)
        
            df_metrics = evaluator.get_metrics(self, data="val")
            val_avg_rmse = df_metrics['val_rmse'].mean()
            
            return val_avg_rmse
            
        return hp_trial

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################

    def get_slices(self, 
                   idx: int, 
                   lengths: np.array,
                   data_type: str
                   )->tuple:
        """
            Returns 2 indices to select the idx-th unit.
            Beware that due to data preprocess, the indices of time series do not correspond to the original units'.

            Parameters
            ----------  
            idx : int
                The index of the fold's original time series. 

            lengths : np.array
                Lengths of each unit.

            data_type : str
                Is either "train", "val", "test" or "test_full".

            Returns
            ----------
            slice_inf, slice_sup : tuple
                The indices to select the desired unit.
        """
        if data_type=="test":
            return idx, idx+1
        else:
            len_list = [0]+list(np.cumsum(np.array(lengths)-self._WINDOW+1))
            slice_inf, slice_sup = int(len_list[idx]), int(len_list[idx+1])
            return slice_inf, slice_sup


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
                Whether the trajectory is in the test dataset or not.

            Returns
            ----------
            cycles : np.array
                The time steps associated to the unit.
        """
        return np.arange(self._WINDOW, length+1) * (10 if self._dataset_name=="femto" else 1)

    def save_hp_values(self,
                       path: str
                       )->None:
        """
            Saves the current hyperparameter values.

            Parameters
            ----------
            path : str
                The path to save the hyperparameter values.
        """
        hp_names = ["LEARNING_RATE"]
        hp_vals = [self._LEARNING_RATE]
        pd.Series({hp_names[i]:hp_vals[i] for i in range(len(hp_names))}).to_csv(path+"_hp_vals.csv")

    # def denorm(self,
    #            y: np.array,
    #            data_type: str,
    #            k : int,
    #            lengths : list
    #            )->np.array:
    #     """
    #          UNUSED. Denormalizes the array of RUL values y, knowing its data type.
    
    #             Parameters
    #             ----------
    #             y : np.array
    #                 The input array of normalized RUL values.

    #             data_type : str
    #                 The type of the dataset of y. Is either "train", "val", "test" or "test_full".

    #             k : int
    #                 The fold index in case of "train" or "val" data_type.

    #             lengths : list
    #                 The original length of each unit.
    #     """
    #     # DeNormRUL(t) = time * NormRUL(t) / (1-NormRUL(t)), where time starts at t0=1 and NormRUL(t0)=(T-t0)/T < 1
    #     is_test = data_type == "test"
    #     return np.vstack([self.get_cycles(l, is_test=is_test) for l in lengths]) * y/(1-y)
  
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
        res = pd.DataFrame({"parameter": ["DATASET_NAME",
                                          "FOLDS",
                                          "VARS_TO_KEEP",
                                          "R_EARLY",
                                          "WINDOW",
                                          "BATCH_SIZE",
                                          "EPOCHS",
                                          "LEARNING_RATE",
                                          "SEED"], 
              "value":[self._dataset_name,
                      self._FOLDS,
                      self._VARS_TO_KEEP,
                      self._R_EARLY,
                      self._WINDOW,
                      self._BATCH_SIZE,
                      self._EPOCHS,
                      self._LEARNING_RATE,
                      self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          False,
                          False,
                          False,
                          True,
                          False]})
        res.to_csv(path+"_config.csv")