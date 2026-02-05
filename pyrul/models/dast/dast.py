# dast.py

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from .dast_core import DASTNet
from sklearn.linear_model import LinearRegression as LR
from pyrul.models.base import RULModel
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars

class DAST_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Dual Aspect Self-Attention based on Transformer for Remaining Useful Life Prediction" (Zhizheng Zhang, 2022).
    """

    def __init__(self, 
                 DATASET_NAME: str,
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = None,
                 R_EARLY: float = 125,
                 WINDOW: int = 30,
                 WINDOW_DEC: int = 5,
                 BATCH_SIZE: int = 256,
                 EPOCHS: int = 100,
                 LEARNING_RATE: float = 1e-3,
                 NB_ENC: int = 1,
                 NB_DEC: int = 1,
                 HEADS: int = 4,
                 NB_NEURONS_VAL: int = 64,
                 DROPOUT: float = None,
                 SEED: int = None
                 )->None:
        """
            DATASET_NAME : str
                The name of the dataset to use.
            
            FOLDS : int
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            R_EARLY : float, default=125
                The RUL threshold to apply to the train and validation labels. If None, no threshold is applied.

            WINDOW : int, default=30
                The size of the segments as input to the prediction model.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depeding on DATASET_NAME.

            BATCH_SIZE : int
                The size of each batch for the data.

            EPOCHS : int
                The number of epochs to train the model.

            LEARNING_RATE : float
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
        self._WINDOW_DEC = WINDOW_DEC
        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._BATCH_SIZE = BATCH_SIZE
        self._EPOCHS = EPOCHS
        self._LEARNING_RATE = LEARNING_RATE
        self._NB_ENC = NB_ENC
        self._NB_DEC = NB_DEC
        self._HEADS = HEADS
        self._NB_NEURONS_VAL = NB_NEURONS_VAL
        self._DROPOUT = DROPOUT

        # Data preprocess metrics
        # Min-max scaling data for raw data, before window sliding
        self._df_train_mins_0 = [None]*self._FOLDS
        self._df_train_maxs_0 = [None]*self._FOLDS
        self._df_train_min_0 = None
        self._df_train_max_0 = None

        # Min-max scaling data for statistical feature extraction of windows
        self._df_train_mins_1 = [None]*self._FOLDS
        self._df_train_maxs_1 = [None]*self._FOLDS
        self._df_train_min_1 = None
        self._df_train_max_1 = None

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
            
        if data_type=="train":
            # Min-max scaling data on raw data
            self._df_train_mins_0[k] = df[self._VARS_TO_KEEP].min()
            self._df_train_maxs_0[k] = df[self._VARS_TO_KEEP].max()
        
        # Step 1. Piecewise Linear RUL labels
        apply_thres = data_type in ["train","val"]
        if apply_thres and self._R_EARLY is not None:
            df["RUL"] = np.minimum(self._R_EARLY, df["RUL"])

            # UNUSED Step 2. RUL normalization w.r.t. R_EARLY
            # df["RUL"] /= self._R_EARLY
    
        # Step 3. Feature selection and Max-norm scaling (except for "unit" and "RUL")
        df_info = df[["unit","RUL"]].copy()
        df = df[self._VARS_TO_KEEP]

        if k!=-1:
            if (self._df_train_mins_0[k] is not None) and (self._df_train_maxs_0[k] is not None):
                df_train_min_0 = self._df_train_mins_0[k]
                df_train_max_0 = self._df_train_maxs_0[k]
            else:
                raise Exception("Data metrics are not built yet. Fit before.")
        else:
            if (self._df_train_min_0 is not None) and (self._df_train_max_0 is not None):
                df_train_min_0 = self._df_train_min_0
                df_train_max_0 = self._df_train_max_0
            else:
                raise Exception("Data metrics are not built yet. Load or fit before.")

        df = (df - df_train_min_0) / (df_train_max_0 - df_train_min_0)
        df[["unit","RUL"]] = df_info
    
        # Step 4. Individual pre-processing
        keep_last = data_type == "test"
        X, y, lengths = self.get_windows_and_labels(df, keep_last=keep_last)

        # Step 5. Min-max scaling on all new features
        new_vars = ["reg_coef", "avg"]
        xfea_min = np.min(X[:,-2:,:], axis=(0,2))
        xfea_max = np.max(X[:,-2:,:], axis=(0,2))

        if data_type=="train":
            # Min-max scaling data on new features
            self._df_train_mins_1[k] = pd.Series(index=new_vars, data=xfea_min)
            self._df_train_maxs_1[k] = pd.Series(index=new_vars, data=xfea_max)

        if k!=-1:
            if (self._df_train_mins_1[k] is not None) and (self._df_train_maxs_1[k] is not None):
                df_train_min_1 = self._df_train_mins_1[k]
                df_train_max_1 = self._df_train_maxs_1[k]
            else:
                raise Exception("Data metrics are not built yet. Fit before.")
        else:
            if (self._df_train_min_1 is not None) and (self._df_train_max_1 is not None):
                df_train_min_1 = self._df_train_min_1
                df_train_max_1 = self._df_train_max_1
            else:
                raise Exception("Data metrics are not built yet. Load or fit before.")

        X[:,-2,:] = (X[:,-2,:] - df_train_min_1["reg_coef"]) / (df_train_max_1["reg_coef"] - df_train_min_1["reg_coef"])
        X[:,-1,:] = (X[:,-1,:] - df_train_min_1["avg"]) / (df_train_max_1["avg"] - df_train_min_1["avg"])
        X = tf.constant(X, dtype=tf.float32)
    
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
        self._df_train_min_0.to_csv(path+"_df_train_min_0.csv")
        self._df_train_max_0.to_csv(path+"_df_train_max_0.csv")
        self._df_train_min_1.to_csv(path+"_df_train_min_1.csv")
        self._df_train_max_1.to_csv(path+"_df_train_max_1.csv")

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
        self._df_train_min_0 = pd.read_csv(path+"_df_train_min_0.csv", sep=",", header=0, index_col=0)['0']
        self._df_train_max_0 = pd.read_csv(path+"_df_train_max_0.csv", sep=",", header=0, index_col=0)['0']
        self._df_train_min_1 = pd.read_csv(path+"_df_train_min_1.csv", sep=",", header=0, index_col=0)['0']
        self._df_train_max_1 = pd.read_csv(path+"_df_train_max_1.csv", sep=",", header=0, index_col=0)['0']
        

    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        self._df_train_min_0 = pd.concat(self._df_train_mins_0, axis=1).mean(axis=1)
        self._df_train_max_0 = pd.concat(self._df_train_maxs_0, axis=1).mean(axis=1)
        self._df_train_min_1 = pd.concat(self._df_train_mins_1, axis=1).mean(axis=1)
        self._df_train_max_1 = pd.concat(self._df_train_maxs_1, axis=1).mean(axis=1)

    def get_windows_and_labels(self, 
                               ds: pd.DataFrame,
                               keep_last: bool=False,
                               extrapolate_test: bool=False,
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

            extrapolate_test : bool, default=False (UNUSED)
                Whether to extend the test trajectory by B-spline extrapolatation per feature, when its length is shorter than the window size.
                Warning : the RUL label will not change.
                Should only be used with keep_last=True.
                
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
        base_x = np.expand_dims(range(self._WINDOW),axis=-1)
        
        for unit in ds['unit'].unique():
            x = ds[ds['unit']==unit].copy()

            # Extrapolation by B-spline to extend test data that is smaller than the desired input length (UNUSED)
            if extrapolate_test and len(x)<self._WINDOW:
                old_ruls = x['RUL'].values
                for v in self._VARS_TO_KEEP: # extrapolation per feature
                    x1 = np.linspace(0, self._WINDOW - 1, len(x))
                    x_new = np.linspace(0, self._WINDOW - 1, self._WINDOW)
                    tck = interpolate.splrep(x1, x[v])
                    a = interpolate.splev(x_new, tck)
                    x.loc[:self._WINDOW,v] = a.tolist() # extending data
                x.loc[:,'RUL'] = np.pad(old_ruls, self._WINDOW-old_len, mode='constant', constant_values=old_ruls[-1])
                
            # Overlapping window slicing
            if keep_last:
                windows = [x[self._VARS_TO_KEEP].to_numpy()[-self._WINDOW:]]
                labels = x['RUL'].iloc[-1:]
                
            else:
                windows = [x[self._VARS_TO_KEEP].to_numpy()[k:k+self._WINDOW] for k in range(len(x)-self._WINDOW+1)]
                # UNUSED Add statistical features
                # windows = []
                # for k in range(len(x)-self._WINDOW+1): # converts the time series of 1 sample into a list of windows
                #     w = x[self._VARS_TO_KEEP].to_numpy()[k:k+self._WINDOW]
                #     fea1 = np.hstack([LR().fit(X=base_x,y=w[:,k]).coef_ for k in range(len(self._VARS_TO_KEEP))]) # (nb_vars) values
                #     fea2 = np.mean(w, axis=0) # (nb_vars) values
                #     w = np.vstack([w, fea1, fea2]) # new window shape : (WINDOW+2, nb_vars)
                #     windows.append(w)
                    
                labels = x['RUL'].iloc[self._WINDOW-1:] # related labels (1 label = window's last cycle's RUL value)
            
            # Adding samples, labels and lengths
            X += windows
            y = np.concatenate([y, labels]) # labels related to each window
            lengths.append(len(x)) # lengths of original time serie
        
        # Final formatting
        X = np.array(X) # final shape : (batch, window+2, features)
        y = tf.constant(y, dtype=tf.float32)
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
        self._models = [DASTNet(dim_val_s = self._NB_NEURONS_VAL, 
                                dim_attn_s = self._NB_NEURONS_VAL, 
                                dim_val_t = self._NB_NEURONS_VAL,
                                dim_attn_t = self._NB_NEURONS_VAL,
                                dim_val = self._NB_NEURONS_VAL, 
                                dim_attn = self._NB_NEURONS_VAL,
                                dec_seq_len = self._WINDOW_DEC, 
                                n_decoder_layers = self._NB_DEC, 
                                n_encoder_layers = self._NB_ENC,
                                n_heads = self._HEADS, 
                                dropout = self._DROPOUT) for k in range(self._FOLDS)]
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
        
            lr_factor = 3*trial.suggest_int("learning_rate_factor", 0, 2)
            lr_factor += (lr_factor==0)
            lr_exp = trial.suggest_int("learning_rate_exp", 1, 5)
            self._LEARNING_RATE = lr_factor * 10**(-lr_exp)
            self._WINDOW_DEC = trial.suggest_int("window_dec", 5, self._WINDOW//2, step=5)
            self._HEADS = trial.suggest_int("heads", 1, 3)
            self._NB_NEURONS_VAL = 2**trial.suggest_int("nb_neurons_val_exp", 4, 5)
            self._NB_ENC = trial.suggest_int("nb_enc", 1, 4)
            self._NB_DEC = trial.suggest_int("nb_dec", 1, 4)
        
            set_seed(self._SEED)

            dataloader.build_train_val_data(self, data_path=data_load_path)

            self._df = None
            self.build_model()
            self.fit_model(verbose=False)
            self._train_datasets = None
        
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
        return np.arange(self._WINDOW, length+1) * (10 if self._dataset_name == "femto" else 1)

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
        hp_names = ["LEARNING_RATE", "WINDOW_DEC", "HEADS", "NB_NEURONS_VAL", "NB_ENC", "NB_DEC"]
        hp_vals = [self._LEARNING_RATE, self._WINDOW_DEC, self._HEADS, self._NB_NEURONS_VAL, self._NB_ENC, self._NB_DEC]
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
                                          "WINDOW",
                                          "WINDOW_DEC",
                                          "BATCH_SIZE",
                                          "EPOCHS",
                                          "LEARNING_RATE",
                                          "NB_ENC",
                                          "NB_DEC",
                                          "HEADS",
                                          "NB_NEURONS_VAL",
                                          "DROPOUT",
                                          "SEED"], 
              "value":[self._dataset_name,
                       self._FOLDS,
                       self._VARS_TO_KEEP,
                       self._R_EARLY,
                       self._WINDOW,
                       self._WINDOW_DEC,
                       self._BATCH_SIZE,
                       self._EPOCHS,
                       self._LEARNING_RATE,
                       self._NB_ENC,
                       self._NB_DEC,
                       self._HEADS,
                       self._NB_NEURONS_VAL,
                       self._DROPOUT,
                       self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          False,
                          True,
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
