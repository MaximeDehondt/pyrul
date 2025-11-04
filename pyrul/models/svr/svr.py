# tsbpdtw.py
from tqdm import tqdm
import numpy as np
import pandas as pd
from pyrul.models.base import RULModel
from sklearn.linear_model import LinearRegression as LR
from sklearn.svm import SVR
from scipy.signal import savgol_filter
import pickle
import itertools
from itertools import combinations
import copy
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars, get_min_ts_size
from pyrul.evaluation.metrics import rmse

class SVR_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Direct Remaining Useful Life Estimation Based on Support Vector Regression" (Racha Khelif, 2016)
    """

    def __init__(self,
                 DATASET_NAME: str,
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = ['T24','T30','T50','Nf','Ps30'],
                 WINDOW: int = 5,
                 SG_WINDOW: int = 31,
                 GAMMA: float = 1,
                 C: float = 10,
                 EPSILON: float = 1e-3,
                 SEED: int = None
                )->None:
        """
            Parameters
            ----------
            DATASET_NAME : str
                The name of the dataset to use.
            
            FOLDS : int, default=None
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=['T24','T30','T50','Nf','Ps30']
                The list of variables to keep. If None, default values will apply depeding on DATASET_NAME.

            WINDOW : int, default=5
                The size of the window for the input data.

            SG_WINDOW : int, default=31
                The window size for the Savitsky-Golay filter of each features.

            GAMMA : float, default=1.
                The gamma parameter of the SVR.

            C : float, default=10
                The regularization parameter C of the SVR.

            EPSILON : float, default=1e-3
                The margin tolerance parameter of the SVR.

            SEED: int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME, 
                         dataset_ori="cmapss_fd001",
                         folds=FOLDS, 
                         seed=SEED)

        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._WINDOW = WINDOW
        self._SG_WINDOW = SG_WINDOW
        self._GAMMA = GAMMA
        self._C = C
        self._EPSILON = EPSILON

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
        do_overlap = data_type in ["test","test_full"]
        keep_last = data_type == "test"

        # Step 1. Selecting initial features
        df = df[["unit","RUL"]+self._VARS_TO_KEEP]
        
        # Step 2. Noise filter
        df = self.apply_SG_filter(df.copy())

        df_info = df[["unit","RUL"]].copy()
        df.drop(columns=["unit","RUL"], inplace=True)

        # Step 3. Min-max norm
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

        df = (df - df_train_min) / (df_train_max - df_train_min)
        df[["unit","RUL"]] = df_info

        # Step 4. Individual pre-processing
        X, y, lengths = self.get_windows_and_labels(df, keep_last=keep_last, do_overlap=do_overlap)
        
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
        train_datasets = self.get_dict_dataset(train_datasets, is_folded=True)
        val_datasets = self.get_dict_dataset(val_datasets, is_folded=True)
        self._train_datasets = train_datasets
        self._val_datasets = val_datasets

        X_train_old = copy.deepcopy(train_datasets["fold_data"])
        X_val_old = copy.deepcopy(val_datasets["fold_data"])
 
        # Feature wrapper selection 
        # Creating a list containing all possible sorted combinations of sensors indices
        # Eg: with 3 sensors : [(0), (1), (2), (0,1), (0,2), (1,2), (0,1,2)] -> cmapss
        # Eg: [(0), (0,1), (0,1,2)] -> femto
        nb_sensors = len(self._VARS_TO_KEEP)

        if self._dataset_name=="femto":
            features_sets = [np.arange(0,k,dtype=int) for k in range(1,nb_sensors)]
        else:
            features_sets = []
            for k in range(1,nb_sensors+1):
                features_sets += list(combinations(list(range(nb_sensors)), k))

        # Iterating to find the best features set yielding the best validation rmse
        best_models = []
        best_val_metric = -np.inf
        best_idx = -1

        for i in tqdm(range(len(features_sets)), desc="feature selection"):
            feature_set = list(features_sets[i])
            features_idx = self.get_features_idx_set(feature_set)

            for k in range(self._FOLDS):
                self._train_datasets["fold_data"][k] = X_train_old[k][:,features_idx]
                self._val_datasets["fold_data"][k] = X_val_old[k][:,features_idx]

            set_seed(self._SEED)
            self.build_model()
            self.fit_model()

            res = []
            for k in range(self._FOLDS):
                X_val, y_val = self._val_datasets["fold_data"][k], self._val_datasets["fold_labels"][k]
                y_val_pred = self._models[k].predict(X_val)
                res.append(rmse(y_val, y_val_pred))
            val_metric = np.mean(res, axis=0)

            # Keep best so far
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_idx = i
                best_models = copy.deepcopy(self._models)
        
        # Keeping the best features set and transforming the training datatset
        self._best_fs = list(features_sets[i])
        features_idx = self.get_features_idx_set(feature_set)

        for k in range(self._FOLDS):
            self._train_datasets["fold_data"][k] = X_train_old[k][:,features_idx]
        self._val_datasets["fold_data"] = X_val_old

        return self._train_datasets

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
        # Converting to dict form
        datasets = self.get_dict_dataset(datasets, is_folded=is_folded)

        # Feature selection
        features_idx = self.get_features_idx_set(self._best_fs)
        if is_folded:
            for k in range(self._FOLDS):
                datasets["fold_data"][k] = datasets["fold_data"][k][:,features_idx]
        else:
            datasets["data"] = datasets["data"][:,features_idx]
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
        self._df_train_min.to_csv(path+"_df_train_min.csv", sep=",")
        self._df_train_max.to_csv(path+"_df_train_max.csv", sep=",")
        np.savetxt(path+"_best_fs.txt", self._best_fs)

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
        self._best_fs = list(np.loadtxt(path+"_best_fs.txt").astype(int))

    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        self._df_train_min = pd.concat(self._df_train_mins, axis=1).mean(axis=1)
        self._df_train_max = pd.concat(self._df_train_maxs, axis=1).mean(axis=1)

    def get_windows_and_labels(self, 
                               ds: pd.DataFrame,
                               do_overlap: bool=False,
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
        # Time-window creation (assuming every lifetime is >= WINDOW)
    
        # Creating training dataset
        X, y, lengths = [], [], []
        unit_list = ds['unit'].unique()
        
        for unit in unit_list:
            x = ds[ds['unit']==unit].copy()

            if keep_last:
                k = len(x)-self._WINDOW
                windows = [x[self._VARS_TO_KEEP].to_numpy()[k:self._WINDOW+k]]
                labels = [x['RUL'].iloc[self._WINDOW+k-1]]
            else:
                if do_overlap:
                    # Overlapping window slicing
                    windows = [x[self._VARS_TO_KEEP].to_numpy()[k:self._WINDOW+k] for k in range(len(x)-self._WINDOW+1)]
                    labels = [x['RUL'].iloc[self._WINDOW+k-1] for k in range(len(x)-self._WINDOW+1)]
                else:
                    # Non-overlapping window slicing
                    windows = [x[self._VARS_TO_KEEP].to_numpy()[self._WINDOW*k:self._WINDOW*(k+1)] for k in range(len(x)//self._WINDOW)]
                    labels = [x['RUL'].iloc[self._WINDOW*(k+1)-1] for k in range(len(x)//self._WINDOW)]
                                
            # Keep average and trend of each window only
            # current windows[i] shape : (WINDOW, nb_vars)
            base_x = np.expand_dims(range(self._WINDOW),axis=-1)       
            windows = [np.hstack(
                                [
                                    np.mean(w,axis=0), 
                                    np.hstack([LR().fit(X=base_x,y=w[:,k]).coef_ for k in range(len(self._VARS_TO_KEEP))])
                                ]
                                )
                      for w in windows]
            # new windows[i] shape : (2*nb_vars) where we have all mean values followed by all trend values
            
            # Adding samples, labels and lengths
            X += windows
            y = np.concatenate([y, labels]) # labels related to each window
            lengths.append(len(x)) # lengths of original time serie
        
        # Final formatting
        X = np.array(X)
        y = np.array(y)
        
        return X, y, lengths

    def apply_SG_filter(self, 
                        ds : pd.DataFrame
                        )->pd.DataFrame:
        """
            Applies the Savitsky-Golay filter to the dataset's units, for each features.
    
            Parameters
            ----------    
            ds: pd.DataFrame
                The dataset. Must have "unit" as a column.  

            Returns
            ----------    
            ds: pd.DataFrame
                The new dataset. 
        """
        unit_list = ds['unit'].unique()
        
        for var in self._VARS_TO_KEEP: 
            for e in unit_list:
                x = ds[ds['unit']==e][var]
                ds.loc[x.index, var] = savgol_filter(x.values, 
                                                    window_length=self._SG_WINDOW, 
                                                    polyorder=2, 
                                                    mode='interp')
        return ds
            
    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        self._models = [SVR(kernel='rbf', 
                            gamma=self._GAMMA, 
                            C=self._C, 
                            epsilon=self._EPSILON) for k in range(self._FOLDS)]

    def fit_model(self)->None:
        """
            Fits each model to their respective training fold.

            Parameters
            ----------
            verbose : bool, default=True
                Whether to display information on the fitting process.
        """
        for k in tqdm(range(self._FOLDS), desc="fitting models"):
            X_train, y_train = self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k]
            self._models[k].fit(X=X_train, y=y_train)

    def predict(self, 
                x: np.array, 
                k: int, 
                do_mean: bool=True,
                is_test: bool=False)->np.array:
        """
            Returns the prediction associated to the input x, for the idxK-th model.
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
        if k!=-1:
            raw_pred = self._models[k].predict(x)
        else:
            preds = [self._models[k].predict(x) for k in range(self._FOLDS)]
            if do_mean:
                raw_pred = np.mean(preds, axis=0)
            else:
                raw_pred = preds
                
        return raw_pred
        
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
        
            self._GAMMA = np.around(trial.suggest_float("SVR_gamma", 0.01, 100), 2)
            C_EXP = trial.suggest_int("SVR_C_exp", -3, 3)
            EPSILON_FACT = trial.suggest_int("SVR_epsilon_fact", 1, 99)
            EPSILON_EXP = trial.suggest_int("SVR_epsilon_exp", -3, 3)
            self._C = 10**C_EXP
            self._EPSILON = EPSILON_FACT * 10**EPSILON_EXP
            self._WINDOW = trial.suggest_int("L", 5, 30, step=5)

            max_sg_window = get_min_ts_size(self._dataset_name)
            self._SG_WINDOW = trial.suggest_int("SG_window", 3, max_sg_window, step=2)
        
            set_seed(self._SEED) # doesn't affect sampler's state

            dataloader.build_train_val_data(self, data_path=data_load_path)
            
            df_metrics = evaluator.get_metrics(self, data="val")
            val_avg_rmse = df_metrics['val_rmse'].mean()
            
            return val_avg_rmse
            
        return hp_trial

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
            with open(path+'model_'+str(k)+'.pkl', 'wb') as f:
                pickle.dump(self._models[k], f)

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
        self._models = []
        for k in range(self._FOLDS):
            with open(path+'model_'+str(k)+'.pkl', 'rb') as f:
                self._models.append(pickle.load(f))

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################

    def get_features_idx_set(self, l):
        """
            Converts the sensor index list into the column indices to choose in X.
            Eg: 
            With N = nb of sensors
            [0] -> [0,N] (mean of sensor 0 + trend of sensor 0)
            [0,2] -> [0,2,N,2+N] (mean of sensor 0 and 2 + trend of sensor 0 and 2)
        """
        N = len(self._VARS_TO_KEEP)
        idx = [i for i in l] + [N+i for i in l]
        return idx

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
            if data_type=="test_full":
                len_list = [0]+list(np.cumsum(np.array(lengths)-self._WINDOW+1))
            else:
                len_list = [0]+list(np.cumsum(np.array(lengths)//self._WINDOW))
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
        if is_test:
            return np.arange(self._WINDOW, length+1) * (10 if self._dataset_name=="femto" else 1)
        else:
            return np.arange(self._WINDOW, length+1, self._WINDOW) * (10 if self._dataset_name=="femto" else 1)

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
        hp_names = ["GAMMA", "C", "EPSILON", "SG_WINDOW", "L"]
        hp_vals = [self._GAMMA, self._C, self._EPSILON, self._SG_WINDOW, self._WINDOW]
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
                                          "SG_WINDOW",
                                          "GAMMA",
                                          "C",
                                          "EPSILON",
                                          "SEED"], 
              "value":[self._dataset_name,
                      self._FOLDS,
                      self._VARS_TO_KEEP,
                      self._WINDOW,
                      self._SG_WINDOW,
                      self._GAMMA,
                      self._C,
                      self._EPSILON,
                      self._SEED], 
              "is_tuned":[False,
                         False,
                         False,
                         True,
                         True,
                         True,
                         True,
                         True,
                         False]})
        res.to_csv(path+"_config.csv") 