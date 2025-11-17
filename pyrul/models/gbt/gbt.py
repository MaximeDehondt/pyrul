# gbt.py

import numpy as np
import pandas as pd
from pyrul.models.base import RULModel
from pyrul.evaluation.metrics import rmse
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.tree import DecisionTreeRegressor as DTR
import copy
import pickle
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars, get_segment_len
from tqdm import tqdm

class GBT_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Remaining useful life (rul) prediction of rolling element bearing using random forest and gradient boosting technique" (Sangram Patil, 2018)
    """

    def __init__(self,
                 DATASET_NAME: str,
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = None,
                 SEGMENT_LEN: int = None,
                 LEARNING_RATE: float = 1e-3,
                 N_ESTIMATORS: int = 100,
                 MIN_SAMPLES_SPLIT: int = 2,
                 MIN_SAMPLES_LEAF: int = 1,
                 MAX_DEPTH: int = 3,
                 WINDOW: int=5,
                 SEED: int = None
                 )->None:
        """
            Parameters
            ----------
            DATASET_NAME : str
                The name of the dataset to use.

            FOLDS : int, default=None
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depending on DATASET_NAME.

            SEGMENT_LEN : int, default=None
                The length of any segment in a unit. If None, default values will apply depending on DATASET_NAME.

            LEARNING_RATE : float
                The learning rate of the boosting algorithm.

            N_ESTIMATORS : int
                The number of boosting stages to perform.

            MIN_SAMPLES_SPLIT : int
                The minimum number of samples required to split an internal node.

            MIN_SAMPLES_LEAF : int
                The minimum number of samples required to be at a leaf node.

            MAX_DEPTH : int
                Maximum depth of the individual regression estimators.

            WINDOW : int, default=30
                The size of the window for the input data when DATASET_NAME=="cmapss_fd001".

            SEED: int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME,
                         dataset_ori="femto",
                         folds=FOLDS,
                         seed=SEED)

        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "femto") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._LEARNING_RATE = LEARNING_RATE
        self._N_ESTIMATORS = N_ESTIMATORS
        self._MIN_SAMPLES_SPLIT = MIN_SAMPLES_SPLIT
        self._MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
        self._MAX_DEPTH = MAX_DEPTH
        self._SEGMENT_LEN = get_segment_len(DATASET_NAME) if SEGMENT_LEN is None else SEGMENT_LEN
        self._WINDOW = WINDOW

        # Data preprocess metrics
        self._best_feature_set = None

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
            
        keep_last = data_type == "test"
    
        if self._dataset_name == "cmapss_fd001":
            # Step 1. Split data into segments
            X, y, lengths = self.get_windows_and_labels(df, keep_last=keep_last)
        
        # Step 2. Extract time-domain features
        elif self._dataset_name == "femto":
        
            # Step 1. Split data into segments
            X, y = self.get_segments_and_labels(df, keep_last=keep_last) # X shape : (segments, time, features)
            lengths = df.groupby("unit", sort=False).count()["RUL"].values // self._SEGMENT_LEN
        
            # Step 2. Extract time-domain features
            X = self.extract_time_features(X)
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])) # collapsing all features together
            
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
        
        # Feature wrapper selection  via RFECV
        # (RFECV modified to avoid "ensemblist" models : each model should have the same hyperparameters on any fold)
        # Step 1. Feature relevance
        if self._dataset_name == "femto":
            var_names = ["X_rms", "Y_rms",
                        "X_kurt", "Y_kurt",
                         "X_max", "Y_max",
                         "X_std", "Y_std",
                         "X_var", "Y_var",
                         "X_skew", "Y_skew",
                         "X_peaktopeak", "Y_peaktopeak",
                         "X_min", "Y_min"]
        elif self._dataset_name == "cmapss_fd001":
            var_names = self._VARS_TO_KEEP
            
        df_ranks = pd.DataFrame({"name": var_names, "rank": np.zeros(len(var_names))}) # best rank=0, worst rank=len(var_names)-1
        features_set = pd.DataFrame({"name": var_names, "init_idx": np.arange(0, len(var_names))}) # init_idx used for original feature indexing in datasets
        # We progressively remove features from features_set, and fill df_ranks
        for q in range(len(var_names)-1):
            
            # Computing average features importance over all fold for current feature set
            fi_vals = []
            for k in range(self._FOLDS):
                dtr = DTR()
                fs_idx = self.get_features_idx_set(features_set.init_idx.values)
                X_train, y_train = self._train_datasets["fold_data"][k][:,fs_idx], self._train_datasets["fold_labels"][k]
                dtr.fit(X_train,y_train)
                fi_vals.append(self.average_fi(L=dtr.feature_importances_, n=len(features_set)))

            fi_avg = np.mean(fi_vals, axis=0)
    
            # Get worst feature
            worst_idx = np.argmin(fi_avg)
            worst_feature = features_set.iloc[worst_idx].name
            worst_init_idx = features_set.iloc[worst_idx].init_idx

            # Set rank and remove feature
            df_ranks.loc[worst_init_idx, "rank"] = len(features_set)-1
            features_set = features_set.drop(index=worst_idx).reset_index(drop=True)
            
        f_ranks = list(df_ranks.sort_values(by="rank", ascending=True).index)

        # Step 2. We search for the overall best feature set iteratively
        res = []
        models_list = []
        for i in range(len(var_names)):
            print("\nTesting",i+1,"features out of",len(var_names),"...")
            fs_idx = self.get_features_idx_set(f_ranks[:i+1])

            for k in range(self._FOLDS):
                self._train_datasets["fold_data"][k] = X_train_old[k][:,fs_idx]
                self._val_datasets["fold_data"][k] = X_val_old[k][:,fs_idx]

            set_seed(self._SEED)
            self.build_model()
            self.fit_model()

            for k in range(self._FOLDS):
                X_val, y_val = self._val_datasets["fold_data"][k], self._val_datasets["fold_labels"][k]
                y_val_pred = self._models[k].predict(X_val)
                err = rmse(y_val, y_val_pred)
                res.append([i, k, err])
                
            models_list.append(self._models)

        df_res = pd.DataFrame(data=res, columns=["features_set","fold","val_err"])
        fs = df_res.groupby(["features_set"], sort=False).mean("val_err")["val_err"].idxmin()

        best_feature_set = np.array(var_names)[np.array(f_ranks)][:fs+1]
        self._best_feature_set = np.array(f_ranks, dtype="int")[:fs+1]

        # Modifying self._models and datasets
        self._models = models_list[fs] # best models

        # Transforming training dataset
        for k in range(self._FOLDS):
            fs_idx = self.get_features_idx_set(self._best_feature_set)
            self._train_datasets["fold_data"][k] = X_train_old[k][:,fs_idx]
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
        datasets = self.get_dict_dataset(datasets, is_folded=is_folded)
        fs_idx = self.get_features_idx_set(self._best_feature_set)

        if is_folded:
            for k in range(self._FOLDS):
                datasets["fold_data"][k] = datasets["fold_data"][k][:,fs_idx]
                
        else:
            datasets["data"] = datasets["data"][:,fs_idx]
            
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
        np.savetxt(path+"_best_feature_set.txt", self._best_feature_set)

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
        self._best_feature_set = np.loadtxt(path+"_best_feature_set.txt", dtype="int")

    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        pass

    def get_segments_and_labels(self, 
                                df: pd.DataFrame, 
                                keep_last: bool=False
                                )->tuple:
        """
            Transforms a dataframe into samples of features and labels arrays.
            Each unit is transformed into several samples.

            Parameters
            ----------
            dataset : pd.DataFrame
                The dataset to convert.
            
            keep_last : bool, default=False
                Whether to keep the last sample of each unit or not.

            Returns
            ----------
            X : np.array
                The feature values of each unit's segments.
            
            y : np.array
                The RUL values of each sample.
        """
        if keep_last:
            sub_df = df.groupby(['unit'], sort=False).tail(self._SEGMENT_LEN) # keep last segment only
            groups = sub_df.groupby(['unit','RUL'], sort=False)
        else:
            groups = df.groupby(['unit','RUL'], sort=False)

        data_series = groups[self._VARS_TO_KEEP].apply(np.array)
        labels = groups['RUL'].head(1).values
    
        X = np.array(data_series.to_list())
        y = labels
                                   
        return X, y

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
        
        for unit in unit_list:
            x = ds[ds['unit']==unit].copy()

            if keep_last:
                k = len(x)-self._WINDOW
                windows = [x[self._VARS_TO_KEEP].to_numpy()[k:self._WINDOW+k]] 
                labels = [x['RUL'].iloc[self._WINDOW+k-1]]
            else:

                # Overlapping window slicing
                windows = [x[self._VARS_TO_KEEP].to_numpy()[k:self._WINDOW+k] for k in range(len(x)-self._WINDOW+1)]
                labels = [x['RUL'].iloc[self._WINDOW+k-1] for k in range(len(x)-self._WINDOW+1)]

            windows = np.array(windows) # shape : (nb_windows, window_size, features)
            
            # Adding samples, labels and lengths
            X += [windows]
            y = np.concatenate([y, labels]) # labels related to each window
            lengths.append(len(x)) # lengths of original time serie
        
        # Final formatting
        X = np.vstack(X)
        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
        y = np.array(y)
        
        return X, y, lengths

    def extract_time_features(self, 
                              x: np.array
                              )->np.array:
        """
            Computes the following time-domain features per sub segment for each unit :
            RMS, Kurtosis, Max, Std, Variance, Skewness, Peak-to-Peak and Min.

            Parameters
            ----------            
            x : np.array 
                The feature dataset array of shape (segments, time, features).

            Returns
            -------
            x_new : np.array
                The new feature dataset array of shape (segments, 11, features).
        """
        mu = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        var = std**2
        
        # RMS
        rms = np.mean(np.square(x), axis=1, keepdims=True)**0.5
    
        # Kurtosis
        kurtosis = np.mean( ((x - mu)/std)**4, axis=1, keepdims=True)
    
        # Max, min of abs
        maxi = np.max(np.abs(x), axis=1, keepdims=True)
        mini = np.min(np.abs(x), axis=1, keepdims=True)
    
        # Skewness
        skew = np.mean( ((x - mu)/std)**3, axis=1, keepdims=True)
    
        # Peak-to-peak
        p2p = np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True)
    
        x_new = np.hstack([rms, kurtosis, maxi, std, var, skew, p2p, mini])

        return x_new
            
    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        self._models = [GBR(learning_rate = self._LEARNING_RATE,
                            n_estimators = self._N_ESTIMATORS,
                            min_samples_split = self._MIN_SAMPLES_SPLIT,
                            min_samples_leaf = self._MIN_SAMPLES_LEAF,
                            max_depth = self._MAX_DEPTH,
                            random_state = self._SEED) for k in range(self._FOLDS)]

    def fit_model(self)->None:
        """
            Fits each model to their respective training fold.
        """
        if self._models is None:
            raise Exception("No models found. Call model.build_model() before.")

        for k in tqdm(range(self._FOLDS), desc="fitting models"):
            X_train, y_train = self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k]
            self._models[k].fit(X=X_train, y=y_train)
        
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
            
            lr_factor = 3*trial.suggest_int('learning_rate_factor', 0, 2)
            lr_factor += (lr_factor==0)
            lr_exp = trial.suggest_int('learning_rate_exp', -3, -1)
            self._LEARNING_RATE = lr_factor * 10**(lr_exp)

            if self._dataset_name=="femto":
                self._N_ESTIMATORS = trial.suggest_int('n_estimators', 0, 200, step=10)
                self._MAX_DEPTH = trial.suggest_int('max_depth', 2, 30)
            else:
                self._N_ESTIMATORS = 10
                self._MAX_DEPTH = trial.suggest_int('max_depth', 2, 10)
            self._N_ESTIMATORS += (self._N_ESTIMATORS==0)
            self._MIN_SAMPLES_SPLIT =  trial.suggest_int('min_samples_split', 2, 100)
            self._MIN_SAMPLES_LEAF =  trial.suggest_int('min_samples_leaf', 1, 100)
        
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
        if self._models is None:
            raise Exception("No models found. Call model.build_model() before.")

        for k in range(self._FOLDS):
            with open(path+'_model_'+str(k)+'.pkl', 'wb') as f:
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
            with open(path+'_model_'+str(k)+'.pkl', 'rb') as f:
                self._models.append(pickle.load(f))

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################
    
    def get_features_idx_set(self, L: list[int])->list[int]:
        """
            Converts the sensor index list into the sorted column indices to choose in X.
            L must be sorted from smallest to biggest index.
            Example : 
            To select sensor 0 and 2, (L=[0,2]), knowing self._WINDOW=3, and the number of variables is 3 :
            Features of data before reshape : [0,1,2] => Features of data after reshape : [0,1,2, 3,4,5, 6,7,8]
            Need to select => [0, 2, 3, 5, 6, 8]

            Parameters
            ----------  
            L : list of int
                The sorted list of variable indices, from smallest to biggest.

            Returns
            ----------
            new_L : list of int
                The sorted list of indices to select the corresponding columns, from smallest to biggest.
        """
        if self._dataset_name == "cmapss_fd001":
            N = len(self._VARS_TO_KEEP)
            idx = []
            for k in range(self._WINDOW):
                idx += [k*N+l for l in L]
            return idx
            
        elif self._dataset_name == "femto":
            return L

    def average_fi(self, L: list[float], n: int)->list[float]:
        """
            Converts the extended feature importances list into a compressed one.
            Example : 
            Features of data after reshape : [0,1,2, 3,4,5, 6,7,8]
            Feature importances : [0.5, 0.1, 0.,   0., 0., 0.1,   0.5, ,0.2, 0.3]
            => Compressed feature importances [0.33, 0.1, 1.33]

            Parameters
            ----------  
            L : list of float
                The extended feature importance values.

            n : int
                The number of initial features, before flattening windows.

            Returns
            ----------
            new_L : list of float
                The compressed feature importance values.
        """
        if self._dataset_name == "femto":
            return L
        elif self._dataset_name == "cmapss_fd001":
            return [np.mean(L[i::n]) for i in range(n)]
    
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
            slice_inf, slice_sup = idx, idx+1
        else:
            w = (1 if self._dataset_name=="femto" else self._WINDOW)
            len_list = [0]+list(np.cumsum(np.array(lengths) - w + 1))
            slice_inf, slice_sup = len_list[idx], len_list[idx+1]
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
        if self._dataset_name == "femto":
            return np.arange(1, length+1) * 10
        elif self._dataset_name == "cmapss_fd001":
            return np.arange(self._WINDOW, length+1)

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
        hp_names = ["LEARNING_RATE", "N_ESTIMATORS", "MIN_SAMPLES_SPLIT", "MIN_SAMPLES_LEAF", "MAX_DEPTH"]
        hp_vals = [self._LEARNING_RATE, self._N_ESTIMATORS, self._MIN_SAMPLES_SPLIT, self._MIN_SAMPLES_LEAF, self._MAX_DEPTH]
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
                                          "SEGMENT_LEN",
                                          "LEARNING_RATE",
                                          "N_ESTIMATORS",
                                          "MIN_SAMPLES_SPLIT",
                                          "MIN_SAMPLES_LEAF",
                                          "MAX_DEPTH",
                                          "WINDOW",
                                          "SEED"], 
              "value":[self._dataset_name,
                       self._FOLDS,
                       self._VARS_TO_KEEP,
                       self._SEGMENT_LEN,
                       self._LEARNING_RATE,
                       self._N_ESTIMATORS,
                       self._MIN_SAMPLES_SPLIT,
                       self._MIN_SAMPLES_LEAF,
                       self._MAX_DEPTH,
                       self._WINDOW,
                       self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          True,
                          (True if self._dataset_name=="femto" else False),
                          True,
                          True,
                          True,
                          False,
                          False]})
        res.to_csv(path+"_config.csv")