# rulrve.py
# Code mostly taken from https://github.com/NahuelCostaCortez/Remaining-Useful-Life-Estimation-Variational
import numpy as np
import pandas as pd
import glob
import pickle
import tensorflow as tf
from pyrul.models.base import RULModel
from .rulrve_core import *
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

class RULRVE_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper "Variational encoding approach for interpretable assessment of remaining
useful life estimation" (Nahuel Costa, 2022)
    """

    def __init__(self, 
                 DATASET_NAME,
                 FOLDS = None,
                 VARS_TO_KEEP = None,
                 WINDOW = 30,
                 ALPHA = 0.1,
                 R_EARLY = 130,
                 LEARNING_RATE = 1e-3,
                 BATCH_SIZE = 128,
                 MAX_EPOCHS = 10000,
                 SEED = None):
        """
            DATASET_NAME : str
                The name of the dataset to use.
            
            FOLDS : int, default=None
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depending on DATASET_NAME.

            WINDOW : int, default=30
                The sequence length.

            ALPHA : float, default=0.1
                The exponentially moving average smoothing alpha parameter.

            R_EARLY : int, default=130
                The RUL threshold to apply to the train and validation labels. If None, no threshold is applied.

            LEARNING_RATE : float, default=1e-3
                The learning rate value for the RUL prediction model.

            BATCH_SIZE : int, default=128
                The batch size for the RUL prediction model.

            MAX_EPOCHS : int, default=10000
                The maximum number of epochs for the RUL prediction model.

            SEED : int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME, 
                         dataset_ori="cmapss_fd001",
                         folds=FOLDS, 
                         seed=SEED)

        if VARS_TO_KEEP is None:
            if DATASET_NAME=="cmapss_fd001":
                self._VARS_TO_KEEP = ["T30","T50","P30","Ps30","phi"]
            else:
                self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001")
                
        self._WINDOW = WINDOW
        self._ALPHA = ALPHA
        self._R_EARLY = R_EARLY
        self._LEARNING_RATE = LEARNING_RATE
        self._BATCH_SIZE = BATCH_SIZE
        self._MAX_EPOCHS = MAX_EPOCHS

        # Data preprocess metrics
        self._df_train_means_stds = [pd.DataFrame({"op_cond":[], "feature":[], "mean":[], "std":[]}) for _ in range(self._FOLDS)]
        self._df_train_mean_std = pd.DataFrame({"op_cond":[], "feature":[], "mean":[], "std":[]})

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
        if self._dataset_name=="femto":
            df['op_cond'] = df["condition"].astype(str)
        else:
            setting_1, setting_2, setting_3 = 'altitude', 'mach', 'seaLvlTemp'
            df[setting_1] = abs(df[setting_1].round())
            df[setting_2] = abs(df[setting_2].round(decimals=2))
        
            df['op_cond'] = df[setting_1].astype(str) + '_' + \
                                df[setting_2].astype(str) + '_' + \
                                df[setting_3].astype(str)

        op_cond = df["op_cond"].unique()
        # Computing mean & std at each newly found operating conditions
        for c in op_cond:
            n = len(self._df_train_means_stds[k])
            self._df_train_means_stds[k] = pd.DataFrame({"op_cond": [c]*len(self._VARS_TO_KEEP), 
                                                         "feature": np.arange(0,len(self._VARS_TO_KEEP)),
                                                         "mean": df[df['op_cond']==c][self._VARS_TO_KEEP].mean().values, 
                                                         "std": df[df['op_cond']==c][self._VARS_TO_KEEP].std().values})

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
        apply_thres = data_type in ["train","val"]
        keep_last = data_type == "test"
        
        # Step 1. Piecewise Linear RUL labels
        if apply_thres and self._R_EARLY is not None:
            df["RUL"] = np.minimum(self._R_EARLY, df["RUL"])

        # Step 2. Adding temporary operating conditions
        if self._dataset_name=="femto":
            df['op_cond'] = df["condition"].astype(str)
        else:
            setting_1, setting_2, setting_3 = 'altitude', 'mach', 'seaLvlTemp'
            df[setting_1] = abs(df[setting_1].round())
            df[setting_2] = abs(df[setting_2].round(decimals=2))
        
            df['op_cond'] = df[setting_1].astype(str) + '_' + \
                                df[setting_2].astype(str) + '_' + \
                                df[setting_3].astype(str)
    
        # Step 3. Feature selection and Max-norm scaling (except for "unit" and "RUL")
        df_info = df[["unit","RUL"]].copy()
        df = df[['op_cond']+self._VARS_TO_KEEP].copy()

        if k!=-1:
            info = self._df_train_means_stds[k]
        else:
            info = self._df_train_mean_std
            
        if (info is not None):
            for c in info["op_cond"].unique():
                df_train_mean = pd.Series(index=self._VARS_TO_KEEP, data=info[info["op_cond"]==c]["mean"].values)
                df_train_std = pd.Series(index=self._VARS_TO_KEEP, data=info[info["op_cond"]==c]["std"].values)
                
                df.loc[df['op_cond']==c, self._VARS_TO_KEEP] = (df.loc[df['op_cond']==c, self._VARS_TO_KEEP] - df_train_mean) / df_train_std
        else:
            raise Exception("Data metrics are not built yet.")

        df.drop(columns=["op_cond"], inplace=True)
        df[["unit","RUL"]] = df_info

        # Step 4. Exponential Smoothing
        df = self.exponential_smoothing(df, self._ALPHA)
    
        # Step 3. Individual pre-processing
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
        self._df_train_mean_std.to_csv(path+"_df_train_mean_std.csv")

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
        self._df_train_mean_std = pd.read_csv(path+"_df_train_mean_std.csv", sep=",")
        

    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        
        for k in range(self._FOLDS):
            self._df_train_means_stds[k]["fold"] = k
        self._df_train_mean_std = pd.concat(self._df_train_means_stds).groupby(["op_cond","feature"], sort=False).mean()

    def get_windows_and_labels(self, 
                               ds: pd.DataFrame,
                               keep_last: bool=False,
                               )->tuple:
        """
            Returns the array of samples, labels and original length, such that each sample is a multi-dimensional window of the initial time series.
    
            Parameters
            ----------
            ds : pd.DataFrame
                The dataset. Must have "unit" and "RUL" as columns at least.

            do_overlap : bool, default=False
                Whether the window slices can overlap or not. Should be 'True' for test data.

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
        X, y, lengths = [], [], []
        unit_list = ds['unit'].unique()
        
        for unit in ds['unit'].unique():
            x = ds[ds['unit']==unit].copy()
    
            # Overlapping window slicing
            if keep_last:
                windows = [x[self._VARS_TO_KEEP].to_numpy()[-self._WINDOW:]]
                labels = x['RUL'].iloc[-1:]
                
            else:
                windows = [x[self._VARS_TO_KEEP].to_numpy()[k:k+self._WINDOW] for k in range(len(x)-self._WINDOW+1)] # converts the time series of 1 sample into a list of windows
                labels = x['RUL'].iloc[self._WINDOW-1:] # related labels (1 label = window's last cycle's RUL value)
    
            # Adding samples, labels and lengths
            X += windows
            y = np.concatenate([y, labels]) # labels related to each window
            lengths.append(len(x)) # lengths of original time serie
        
        # Final formatting
        X = np.array(X)
        y = np.array(y)
        
        X = tf.convert_to_tensor(X) # final shape : (batch, window, features)
        y = tf.expand_dims(y,-1) # final shape : (batch, 1)
        
        return X, y, lengths

    def exponential_smoothing(self, 
                              df: pd.DataFrame,  
                              alpha: float=0.4
                              )->pd.DataFrame:
        """
            Applies an exponential weighted moving average to smooth each features for each unit.

            Parameters
            ----------
            df : pd.DataFrame
                The dataset. Must contain "unit" as a column.

            alpha : float
                The smoothing factor.

            Returns
            -------
            df : pd.DataFrame
                The modified dataset.
        """
        df = df.copy()

        df[self._VARS_TO_KEEP] = df.groupby('unit', sort=False)[self._VARS_TO_KEEP].apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)
        
        return df

    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        self._models = [RVE(timesteps=self._WINDOW, 
                            input_dim=len(self._VARS_TO_KEEP), 
                            masking_value=-99., 
                            intermediate_dim=300, 
                            latent_dim=2, 
                            seed=self._SEED) for k in range(self._FOLDS)]
        optimizers = [tf.keras.optimizers.Adam(learning_rate=0.) for k in range(self._FOLDS)]
        
        for k in range(self._FOLDS):
            if self._train_datasets is not None:
                _ = self._models[k](self._train_datasets["fold_data"][k][:2])
            elif self._test_dataset is not None:
                _ = self._models[k](self._test_dataset["data"][:2])
            else:
                raise AttributeError("Model has no defined train or test datasets to use.")

            self._models[k].compile(optimizer=optimizers[k], loss=tf.keras.losses.MeanSquaredError) # dummy entry for loss

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
                The path to load the models.
        """
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
        # LR finder
        learning_rate = self.find_lr()
        for k in range(self._FOLDS):
            self._models[k].optimizer.learning_rate.assign(learning_rate)
            
        histories = []
        for k in tqdm(range(self._FOLDS), desc="fitting models"):
            X_train, y_train = self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k]
            X_val, y_val = self._val_datasets["fold_data"][k], self._val_datasets["fold_labels"][k]

            model_callbacks = [
        		EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              verbose=1, 
                              patience=30),
                ModelCheckpoint(filepath='RULRVE_model_checkpoints/RULRVE_model_'+str(k)+'.weights.h5',
                                monitor='val_loss', 
                                mode='min', 
                                verbose=verbose, 
                                save_best_only=True, 
                                save_weights_only=True)
                ]
        
            hist = self._models[k].fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 batch_size=self._BATCH_SIZE,
                                 epochs=self._MAX_EPOCHS,
                                 shuffle=True,
                                 callbacks=model_callbacks,
                                 verbose=verbose)
            histories.append(hist)

        # Loading best checkpoints in memory
        self.load_models(path='RULRVE_model_checkpoints/RULRVE')
        
        return histories

    def find_lr(self)->float:
        """
            Finds the optimal learning rate using current models, across all folds.

            Returns
            -------
            best_lr : float
                The optimal learning rate.
        """
        # Train each model for 5 epochs
        # with learning rate growing exponentially from 0.0000001 to 1
        lr_finders = [LRFinder(self._models[k]) for k in range(self._FOLDS)]
        best_lrs = []
        for k in tqdm(range(self._FOLDS), desc="LR Finder"):
            x_train = self._train_datasets["fold_data"][k]
            y_train = self._train_datasets["fold_labels"][k]
            lr_finders[k].find(x_train, y_train, start_lr=0.000001, end_lr=0.1, batch_size=self._BATCH_SIZE, epochs=5)
            best_lrs.append(lr_finders[k].get_best_lr(sma=1))
            
        return np.mean(best_lrs)

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
                The path to load the training and validation data.

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
            self.train_model(verbose=2)
        
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
            Returns 2 indices corresponding to the fold's idx-th unit.
            Beware that because of data preprocess, the index of time series do not correspond
            to the original units' index.

            Parameters
            ----------  
            idx : int
                The index of the fold's original time serie. 

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
                                          "WINDOW",
                                          "ALPHA",
                                          "R_EARLY",
                                          "LEARNING_RATE",
                                          "BATCH_SIZE",
                                          "MAX_EPOCHS",
                                          "SEED"], 
              "value":[self._dataset_name,
                       self._FOLDS,
                       self._VARS_TO_KEEP,
                       self._WINDOW,
                       self._ALPHA,
                       self._R_EARLY,
                       self._LEARNING_RATE,
                       self._BATCH_SIZE,
                       self._MAX_EPOCHS,
                       self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          False,
                          False,
                          False,
                          False,
                          False,
                          False]})
        res.to_csv(path+"_config.csv")
