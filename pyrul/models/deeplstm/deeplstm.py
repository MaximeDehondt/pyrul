# deeplstm.py

import numpy as np
import pandas as pd
import glob
import pickle
import tensorflow as tf
from pyrul.models.base import RULModel
from .deeplstm_core import DeepLSTM
from pyrul.utils.seed import set_seed
from tqdm import tqdm
from pyrul.utils.dataset_info import get_defaults_vars, get_max_ts_size

class DeepLSTM_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Long Short-Term Memory Network for Remaining Useful Life Estimation" (S Zheng, 2017).
    """

    def __init__(self, 
                 DATASET_NAME: str,
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = None,
                 R_EARLY: float = 130,
                 BATCH_SIZE: int = 16,
                 LEARNING_RATE: float = 1e-3,
                 ES: int = 10,
                 EPOCHS: int = 300,
                 LSTM1_N: int = 32,
                 LSTM2_N: int = 64,
                 FNN1_N: int = 8,
                 FNN2_N: int = 8,
                 LSTM1_DP: float = 0.,
                 LSTM2_DP: float = 0.,
                 FNN1_DP: float = 0.,
                 FNN2_DP: float = 0.,
                 L2: int = 10,
                 MAX_TS_LEN: int = None,
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

            BATCH_SIZE : int, default=16
                The size of each batch for the data.

            LEARNING_RATE : float, default=1e-3
                The learning rate for RMSProp to train the model.

            ES : int, default=10
                The early stopping tolerance.

            EPOCHS : int, default=300
                The number of epochs to train the model.

            LSTM1_N : int, default=32
                The number of units in the first LSTM layer.

            LSTM2_N : int, default=64
                The number of units in the second LSTM layer.

            FNN1_N : int, default=8
                The number of units in the first dense layer.

            FNN2_N : int, default=8
                The number of units in the second dense layer. 

            LSTM1_DP : float, default=0.
                The dropout rate in the first LSTM layer.

            LSTM2_DP : float, default=0.
                The dropout rate in the second LSTM layer.

            FNN1_DP : float, default=0.
                The dropout rate in the first dense layer.

            FNN2_DP : float, default=0.
                The dropout rate in the second dense layer.

            L2 : float, default=10.
                The L2-norm penalization value.

            MAX_TS_LEN : int, default=None
                The maximum length of a unit. If None, default values will apply depending on DATASET_NAME.

            SEED : int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME, 
                         dataset_ori="cmapss_fd001",
                         folds=FOLDS, 
                         is_deep_lstm=True,
                         seed=SEED)

        self._R_EARLY = R_EARLY
        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._BATCH_SIZE = BATCH_SIZE
        self._EPOCHS = EPOCHS
        self._LEARNING_RATE = LEARNING_RATE
        self._ES = ES
        self._LSTM1_N = LSTM1_N
        self._LSTM2_N = LSTM2_N
        self._FNN1_N = FNN1_N
        self._FNN2_N = FNN2_N
        self._LSTM1_DP = LSTM1_DP
        self._LSTM2_DP = LSTM2_DP
        self._FNN1_DP = FNN1_DP
        self._FNN2_DP = FNN2_DP
        self._L2 = L2
        self._MAX_TS_LEN = get_max_ts_size(DATASET_NAME) if MAX_TS_LEN is None else MAX_TS_LEN

        # Data preprocess metrics
        self._df_train_means = [None]*self._FOLDS
        self._df_train_stds = [None]*self._FOLDS
        self._df_train_mean = None
        self._df_train_std = None

    ###############################################################
    ##################### DATA PREPROCESSING ######################
    ###############################################################
    
    def data_process_local_fit_and_transform(self, 
                                             df: pd.DataFrame, 
                                             k: int
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
            
        # Z-norm     
        self._df_train_means[k] = df[self._VARS_TO_KEEP].mean()
        self._df_train_stds[k] = df[self._VARS_TO_KEEP].std()

        return self.data_process_local_transform(df=df, k=k, data_type="train")

    def data_process_local_transform(self, 
                                     df: pd.DataFrame,
                                     k: int,
                                     data_type: str,
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
    
        # Step 2. Feature selection and Max-norm scaling (except for "unit" and "RUL")
        df_info = df[["unit","RUL"]].copy()
        df = df[self._VARS_TO_KEEP]

        if k!=-1:
            if (self._df_train_means[k] is not None) and (self._df_train_stds[k] is not None):
                df_train_mean = self._df_train_means[k]
                df_train_std = self._df_train_stds[k]
            else:
                raise Exception("Data metrics are not built yet. Fit before.")
        else:
            if (self._df_train_mean is not None) and (self._df_train_std is not None):
                df_train_mean = self._df_train_mean
                df_train_std = self._df_train_std
            else:
                raise Exception("Data metrics are not built yet. Load or fit before.")

        df = (df - df_train_mean) / df_train_std
        df[["unit","RUL"]] = df_info
    
        # Step 3. Individual pre-processing
        X, y, lengths = self.get_ts_and_labels(df)

        # Step 4. Zero-Padding
        X = tf.keras.utils.pad_sequences(X, maxlen=self._MAX_TS_LEN, dtype='float32', padding='post', truncating='post', value=-1.)
        y = tf.keras.utils.pad_sequences(y, maxlen=self._MAX_TS_LEN, dtype='float32', padding='post', truncating='post', value=-1.)
        X = tf.constant(X)
        y = tf.constant(y)
    
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
        train_datasets = self.data_process_global_transform(datasets=train_datasets, data_type="train")
        return train_datasets

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
        self._df_train_mean.to_csv(path+"_df_train_mean.csv", sep=",")
        self._df_train_std.to_csv(path+"_df_train_std.csv", sep=",")

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
        self._df_train_mean = pd.read_csv(path+"_df_train_mean.csv", sep=",", header=0, index_col=0)['0']
        self._df_train_std = pd.read_csv(path+"_df_train_std.csv", sep=",", header=0, index_col=0)['0']

    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        self._df_train_mean = pd.concat(self._df_train_means, axis=1).mean(axis=1)
        self._df_train_std = pd.concat(self._df_train_stds, axis=1).mean(axis=1)

    def get_ts_and_labels(self, 
                          df: pd.DataFrame
                          )->tuple:
        """
            Returns the complete multivariate time series of each unit, their labels and length.

            Parameters
            ----------
            df : df.DataFrame
                The dataset. Must have "unit" and "RUL" as columns at least.

            Returns
            -------
            X : np.array
                The list of time series of each unit.

            y : np.array
                The associated RUL values.

            lengths : np.array
                The lengths of each unit.
        """
        X, y, lengths = [], [], []
        unit_list = df['unit'].unique()
        
        for unit in df['unit'].unique():
            x = df[df['unit']==unit].copy()
            X += [x[self._VARS_TO_KEEP].to_numpy()]
            y += [np.expand_dims(x['RUL'].values,-1)]
            lengths.append(len(x))
        
        return X, y, lengths

    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        # Building the models
        self._models = [DeepLSTM(lstm1_n=self._LSTM1_N,
                        lstm2_n=self._LSTM2_N,
                        fnn1_n=self._FNN1_N,
                        fnn2_n=self._FNN2_N,
                        lstm1_dp=self._LSTM1_DP,
                        lstm2_dp=self._LSTM2_DP,
                        fnn1_dp=self._FNN1_DP,
                        fnn2_dp=self._FNN2_DP,
                        l2_pen=self._L2) for k in range(self._FOLDS)]
        optimizers = [tf.keras.optimizers.RMSprop(learning_rate=self._LEARNING_RATE) for k in range(self._FOLDS)]
        
        for k in range(self._FOLDS):
            if self._train_datasets is not None:
                _ = self._models[k](self._train_datasets["fold_data"][k][:2])
            elif self._test_dataset is not None:
                _ = self._models[k](self._test_dataset["data"][:2])
            else:
                raise AttributeError("Model has no defined train or test datasets to use.")

            self._models[k].compile(optimizer=optimizers[k], loss=tf.keras.losses.MeanSquaredError)

    def fit_model(self, 
                  verbose: bool=True
                  )->None:
        """
            Fits each model to their respective training fold.

            Parameters
            ----------
            verbose : bool, default=True
                Whether to display information on the fitting process.
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
                                 callbacks=tf.keras.callbacks.EarlyStopping(
                                    monitor='val_mse',
                                    patience=self._ES,
                                    verbose=0,
                                    mode='min'),
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
        
            self._LSTM1_DP = np.around(trial.suggest_float("lstm1_dp", 0., 0.99, step=0.01), 2)
            self._LSTM2_DP =  np.around(trial.suggest_float("lstm2_dp", 0., 0.99, step=0.01), 2)
            self._FNN1_DP =  np.around(trial.suggest_float("fnn1_dp", 0., 0.99, step=0.01), 2)
            self._FNN2_DP =  np.around(trial.suggest_float("fnn2_dp", 0., 0.99, step=0.01), 2)
            lr_factor = 3*trial.suggest_int("learning_rate_factor", 0, 2)
            lr_factor += (lr_factor==0)
            lr_exp = trial.suggest_int("learning_rate_exp", 2, 5)
            self._LEARNING_RATE = lr_factor * 10**(-lr_exp)
            self._L2 =  np.around(trial.suggest_float("l2", 0., 200., step=0.01), 2)
            self._ES = trial.suggest_int("early_stopping", 1, 10)
            self._BATCH_SIZE = 2**trial.suggest_int("batch_size_exp", 3, 7)
        
            set_seed(self._SEED)

            dataloader.build_train_val_data(self, data_path=data_load_path)
            
            self.build_model()
            self.fit_model(verbose=False)
            
            df_metrics = evaluator.get_metrics(self, data="val")
            val_avg_rmse = df_metrics['val_rmse'].mean()
            
            return val_avg_rmse
            
        return hp_trial

    def save_models(self, 
                    path: str
                    )->None:
        """
            Saves the each model's state.

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

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################

    def get_slices(self, 
                   idx: int, 
                   lengths: np.array,
                   data_type: str,
                   is_unmasked: bool=False,
                  )->tuple:
        """
            Returns 2 indices to select the idx-th unit.
            Beware that due to data preprocess, the indices of time series do not correspond to the original units'.

            Parameters
            ----------  
            idx : int
                The index of the fold's original time series. 

            lengths : np.array
                Not used. Lengths of each unit.

            data_type : str
                Is either "train", "val", "test" or "test_full".

            is_unmasked : bool, default=False
                Whether the data has been flattened by unmasking.

            Returns
            ----------
            slice_inf, slice_sup : tuple
                The indices to select the desired unit.
        """
        if is_unmasked:
            len_list = [0]+list(np.cumsum(np.array(lengths)))
            slice_inf, slice_sup = len_list[idx], len_list[idx+1]
        else:
            slice_inf, slice_sup = idx, idx+1
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
        return np.arange(1, length+1)

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
        hp_names = ["LSTM1_DP", "LSTM2_DP", "FNN1_DP", "FNN2_DP", "LEARNING_RATE", "L2", "ES", "BATCH_SIZE"]
        hp_vals = [self._LSTM1_DP, self._LSTM2_DP, self._FNN1_DP, self._FNN2_DP, self._LEARNING_RATE, self._L2, self._ES, self._BATCH_SIZE]
        pd.Series({hp_names[i]:hp_vals[i] for i in range(len(hp_names))}).to_csv(path+"_hp_vals.csv")

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
                                          "BATCH_SIZE",
                                          "LEARNING_RATE",
                                          "ES",
                                          "EPOCHS",
                                          "LSTM1_N",
                                          "LSTM2_N",
                                          "FNN1_N",
                                          "FNN2_N",
                                          "LSTM1_DP",
                                          "LSTM2_DP",
                                          "FNN1_DP",
                                          "FNN2_DP",
                                          "L2",
                                          "MAX_TS_LEN",
                                          "SEED"], 
              "value":[self._dataset_name,
                      self._FOLDS,
                      self._VARS_TO_KEEP,
                      self._R_EARLY,
                      self._BATCH_SIZE,
                      self._LEARNING_RATE,
                      self._ES,
                      self._EPOCHS,
                      self._LSTM1_N,
                      self._LSTM2_N,
                      self._FNN1_N,
                      self._FNN2_N,
                      self._LSTM1_DP,
                      self._LSTM2_DP,
                      self._FNN1_DP,
                      self._FNN2_DP,
                      self._L2,
                      self._MAX_TS_LEN,
                      self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          True,
                          True,
                          True,
                          False,
                          False,
                          False,
                          False,
                          False,
                          True,
                          True,
                          True,
                          True,
                          True,
                          False,
                          False
                          ]})
        res.to_csv(path+"_config.csv")