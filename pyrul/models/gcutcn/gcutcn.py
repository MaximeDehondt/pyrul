# gcutcn.py

from tqdm import tqdm
import numpy as np
import pandas as pd
from pyrul.models.base import RULModel
import pickle
import tensorflow as tf
from pyrul.utils.seed import set_seed
from .gcutcn_core import GCUTCN, OneCycleScheduler
from scipy.fft import rfft, rfftfreq
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from pyrul.utils.dataset_info import get_defaults_vars, get_segment_len

class GCUTCN_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Remaining useful life estimation of bearing via temporal convolutional networks enhanced by a gated convolutional unit" (Yujie Qin, 2024)
    """

    def __init__(self,
                 DATASET_NAME: str,
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = None,
                 NB_FEATURES: int = 32,
                 WINDOW: int = 16,
                 EPOCHS: int = 100,
                 BATCH_SIZE: int = 256,
                 LEARNING_RATE: float = 0.1,
                 ONECLR_MOM_MIN: float = 0.85,
                 ONECLR_MOM_MAX: float = 0.95,
                 ONECLR_PHASE1_PCT: float = 0.3,
                 ONECLR_DIV_FACTOR: float = 25.,
                 SEGMENT_LEN: int = None,
                 SEED: int = None
                 )->None:
        """
            Parameters
            ----------
            DATASET_NAME : str
                The name of the dataset to use.

            FOLDS : int
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depending on DATASET_NAME.

            NB_FEATURES : int
                The final number of features to keep.

            WINDOW : int
                The number of input windows for the model.

            EPOCHS : int
                The number of epochs for the training the model.

            BATCH_SIZE : int
                The number of batch size for training the model.

            LEARNING_RATE : float
                The learning rate of the model optimizer.

            ONECLR_MOM_MIN : float
                The minimum value of the momentum in the OneCycleLR schedule.

            ONECLR_MOM_MAX : float
                The maximum value of the momentum in the OneCycleLR schedule.

            ONECLR_PHASE1_PCT : float
                The percentage of the first phase duration in the OneCycleLR schedule.

            ONECLR_DIV_FACTOR : float
                The division factor in the OneCycleLR schedule.

            SEGMENT_LEN : int
                The length of a segment. If None, default values will apply depending on DATASET_NAME.

            SEED: int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME,
                         dataset_ori="femto",
                         folds=FOLDS, 
                         seed=SEED)

        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "femto") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._NB_FEATURES = NB_FEATURES
        self._WINDOW = WINDOW
        self._EPOCHS = EPOCHS
        self._BATCH_SIZE = BATCH_SIZE
        self._LEARNING_RATE = LEARNING_RATE
        self._SEGMENT_LEN = get_segment_len(DATASET_NAME) if SEGMENT_LEN is None else SEGMENT_LEN
        self._ONECLR_MOM_MIN = ONECLR_MOM_MIN
        self._ONECLR_MOM_MAX = ONECLR_MOM_MAX
        self._ONECLR_PHASE1_PCT = ONECLR_PHASE1_PCT
        self._ONECLR_DIV_FACTOR = ONECLR_DIV_FACTOR

        # Data preprocess metrics
        self._df_train_mins = [None]*self._FOLDS
        self._df_train_maxs = [None]*self._FOLDS
        self._df_train_min = None
        self._df_train_max = None
        self._best_feature_idx = None
        self._feature_weights = [None]*self._FOLDS

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
            
        df = df[self._VARS_TO_KEEP+["unit","RUL"]]
        
        if self._dataset_name=="femto":
            # Step 1. Feature Extraction
            # We need to get raw segments...
            X, y = self.get_ts_and_labels(df, keep_last=False) # X shape : (segments, time, accelerometer)
            
            # ...then extract 11 time-domain features + 17 frequency-domain features...
            var_names = ["f"+str(i) for i in range(1,28*len(self._VARS_TO_KEEP)+1)]
            X1 = self.extract_time_features(X)
            X2 = self.extract_freq_features(X, sampling_freq=25600) 
            X = np.hstack([X1,X2])
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2])) 
            # collapsing all features together : order of feature is acc1_time_feature1, acc2_time_feature1, acc1_time_feature2, acc2_time_feature2, ..., acc1_freq_feature1, acc2_freq_feature1, ...

            last_data = df.groupby(["unit","RUL"], sort=False).tail(1)
            last_ruls, unit_idx = last_data['RUL'].values, last_data['unit'].values

            # ...and build new dataframe for next step.
            new_df = np.hstack([np.expand_dims(unit_idx,-1), np.expand_dims(last_ruls,-1), X])
            new_df = pd.DataFrame(data=new_df)
            new_df.columns = ["unit", "RUL"] + var_names

        elif self._dataset_name=="cmapss_fd001": # no feature extraction for cmapss
            var_names = self._VARS_TO_KEEP
            new_df, y = df.copy(), np.expand_dims(df["RUL"].values,-1)

        # Step 2. Min-max Normalization
        
        # UNUSED # Scaling RUL
        # new_df_gb_rul = new_df.groupby("unit", sort=False)["RUL"]
        # new_df["RUL"] = new_df_gb_rul.transform(lambda x: x/(1+x.iloc[0]))

        # Scaling Features (warning: some features may be constant)
        df_train_max = new_df[var_names].max()
        df_train_min = new_df[var_names].min()
        self._df_train_mins[k] = df_train_min
        self._df_train_maxs[k] = df_train_max

        mask = ((df_train_max - df_train_min).abs()<1e-6)
        cst_vars = list(mask[mask==True].index)
        non_cst_vars = list(mask[mask==False].index)

        new_df[cst_vars] = 0.
        new_df[non_cst_vars] = 2*(new_df[non_cst_vars] - df_train_min[non_cst_vars]) / (df_train_max[non_cst_vars] - df_train_min[non_cst_vars]) - 1

        lengths = new_df.groupby("unit", sort=False).count()["RUL"].values

        return [new_df, y, lengths]

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
        
        df = df[self._VARS_TO_KEEP+["unit","RUL"]]

        if self._dataset_name=="femto":
            # Step 1. Feature Extraction
            # We need to get raw segments...
            X, y = self.get_ts_and_labels(df, keep_last=False) # X shape : (segments, time, accelerometer)
            
            # ...then extract 11 time-domain features + 17 frequency-domain features...
            var_names = ["f"+str(i) for i in range(1,28*len(self._VARS_TO_KEEP)+1)]
            X1 = self.extract_time_features(X)
            X2 = self.extract_freq_features(X, sampling_freq=25600) 
            X = np.hstack([X1,X2])
            X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))

            # collapsing all features together : order of feature is acc1_time_feature1, acc2_time_feature1, acc1_time_feature2, acc2_time_feature2, ..., acc1_freq_feature1, acc2_freq_feature1, ...
            last_data = df.groupby(["unit","RUL"], sort=False).tail(1)
            last_ruls, unit_idx = last_data['RUL'].values, last_data['unit'].values

            # ...and build new dataframe for next step.
            new_df = np.hstack([np.expand_dims(unit_idx,-1), np.expand_dims(last_ruls,-1), X])
            new_df = pd.DataFrame(data=new_df)
            new_df.columns = ["unit", "RUL"] + var_names

        elif self._dataset_name=="cmapss_fd001": # no feature extraction for cmapss
            var_names = self._VARS_TO_KEEP
            new_df, y = df.copy(), np.expand_dims(df["RUL"].values,-1)

        # Step 2. Min-max Normalization
        
        # UNUSED # Scaling RUL
        # new_df_gb_rul = new_df.groupby("unit", sort=False)["RUL"]
        # new_df["RUL"] = new_df_gb_rul.transform(lambda x: x/(1+x.iloc[0]))
        
        # Scaling Features (warning: some features may be constant)
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

        mask = ((df_train_max - df_train_min).abs()<1e-6)
        cst_vars = list(mask[mask==True].index)
        non_cst_vars = list(mask[mask==False].index)
        
        new_df[cst_vars] = 0.
        new_df[non_cst_vars] = 2*(new_df[non_cst_vars] - df_train_min[non_cst_vars]) / (df_train_max[non_cst_vars] - df_train_min[non_cst_vars]) - 1

        lengths = new_df.groupby("unit", sort=False).count()["RUL"].values

        return [new_df, y, lengths]

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
        self._train_datasets = train_datasets

        # Computing best features per fold
        feature_weights = []
        for k in range(self._FOLDS):
            y = self._train_datasets["fold_labels"][k]
            X_vals = self._train_datasets["fold_data"][k].drop(columns=["unit","RUL"]).values
            lengths = self._train_datasets["fold_lengths"][k]

            # Feature relevance
            C = self.feature_indicators(X_vals, y, lengths) # returned shape : (features, 3)
            C = np.nan_to_num(C, nan=0.)
            proba = np.abs(C) / np.sum(np.abs(C), axis=0)
            
            plogp = proba*np.log(proba)
            plogp[proba==0] = 0 # to set p*log(p)=0 when p=0 instead of nans
            
            entropies = (-1/np.log(C.shape[0])) * np.sum(plogp ,axis=0) # shape : (3)
            w = (1-entropies) / np.sum(1-entropies)
            C_all = np.abs(C) @ np.expand_dims(w,-1)
            C_all = C_all[:,0]
            feature_weights.append(C_all) # C_all shape : (features)

        # Keeping _NB_FEATURES best features overall
        avg_feature_weights = pd.Series(np.mean(feature_weights, axis=0))
        self._best_feature_idx =  np.array(avg_feature_weights.sort_values(ascending=False).index[:self._NB_FEATURES], dtype="int")
        self._feature_weights = [softmax([self._best_feature_idx]) for m in feature_weights]
        
        # Initialisation of 1st layer with feature relevance + feature selection + time series extraction
        self.build_model()
        for k in range(self._FOLDS):

            # For training data
            X, y = self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k]

            w_all = self._feature_weights[k]

            # Matrix initialization
            first_layer_W_init = np.outer(w_all, np.ones(64)) # 64 is the chosen nb of neurons in the first dense layer
            first_layer_bias_init = self._models[k].layers[0].linear1.weights[1]
            self._models[k].layers[0].linear1.set_weights([first_layer_W_init, first_layer_bias_init])

            # Feature selection
            new_df = X.drop(columns=["unit","RUL"]).iloc[:, self._best_feature_idx]
            new_df[["unit","RUL"]] = X[["unit","RUL"]]

            # Converting to samples of segments because it would have been harder to deal with a complex dataset earlier for feature selection
            X_pro, y_pro, len_train = self.get_windows_and_labels(new_df, keep_last=False)

            self._train_datasets["fold_data"][k] = X_pro
            self._train_datasets["fold_labels"][k] = y_pro
            self._train_datasets["fold_lengths"][k] = len_train

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
        keep_last = data_type == "test"
        
        datasets = self.get_dict_dataset(datasets, is_folded=is_folded)
        
        if is_folded:
            for k in range(self._FOLDS):
                X, y = datasets["fold_data"][k], datasets["fold_labels"][k]

                # Feature selection
                new_df = X.drop(columns=["unit","RUL"]).iloc[:, self._best_feature_idx]
                new_df[["unit","RUL"]] = X[["unit","RUL"]]
    
                # Converting to samples of segments
                X_pro, y_pro, len_train = self.get_windows_and_labels(new_df, keep_last=keep_last)
    
                datasets["fold_data"][k] = X_pro
                datasets["fold_labels"][k] = y_pro
                datasets["fold_lengths"][k] = len_train

        else:
            X, y = datasets["data"], datasets["labels"]

            # Feature selection
            new_df = X.drop(columns=["unit","RUL"]).iloc[:, self._best_feature_idx]
            new_df[["unit","RUL"]] = X[["unit","RUL"]]
            
            # Converting to samples of segments
            X_pro, y_pro, len_train = self.get_windows_and_labels(new_df, keep_last=keep_last)

            datasets["data"] = X_pro
            datasets["labels"] = y_pro
            datasets["lengths"] = len_train
            
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
        np.savetxt(path+"_best_feature_idx", self._best_feature_idx)

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
        self._best_feature_idx = np.loadtxt(path+"_best_feature_idx", dtype="int")
        
    def compute_avg_local_metrics(self)->None:
        """
            Uses all local data metrics computed from each fold during the creation of the training and validation datasets, to compute an average local metric information.
        """
        self._df_train_min = pd.concat(self._df_train_mins, axis=1).mean(axis=1)
        self._df_train_max = pd.concat(self._df_train_maxs, axis=1).mean(axis=1)

    def get_ts_and_labels(self, 
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
        y = np.expand_dims(labels,axis=-1)
        return X, y

    def extract_time_features(self, 
                              x):
        """
            Computes the following time-domain features per sub segment for each unit :
                - Avg
                - Std
                - RMS
                - Peak-to-Peak
                - Skweness
                - Kurtosis
                - Peak Factor
                - Crest Factor
                - Impulse Factor
                - Margin
                - f11
            
            x : (segments, time, features)
    
            Returns (segments, 11, features)
        """
        # Mean, Std
        f1 = np.mean(x, axis=1, keepdims=True)
        f2 = np.std(x, axis=1, keepdims=True)
    
        # RMS
        f3 = np.mean(np.square(x), axis=1, keepdims=True)**0.5
    
        # Peak-to-peak
        f4 = (np.max(x, axis=1, keepdims=True) - np.min(x, axis=1, keepdims=True))/2
    
        # Skewness
        f5 = np.mean( ((x - f1)/f2)**3, axis=1, keepdims=True)
    
        # Kurtosis
        f6 = np.mean( ((x - f1)/f2)**4 - 3, axis=1, keepdims=True)
    
        # Peak Factor, Crest Factor, Impulse Factor
        f7 = f4/f3
        f8 = f3/f1
        f9 = f4/f1
    
        # Margin
        f10 = np.abs(np.max(x, axis=1, keepdims=True)) / np.mean(np.abs(x)**0.5, axis=1, keepdims=True)**2
    
        # f11
        f11 = np.mean((x - f1)**4, axis=1, keepdims=True) / np.mean(x**2, axis=1, keepdims=True)**2
    
        x_new = np.hstack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])
    
        return x_new

    def extract_freq_features(self, x, sampling_freq):
        """
            Computes the following frequency-domain features per sub segment for each unit :
                - Avg
                - Std
                - Skweness
                - Kurtosis
                - f16
                - f17
                - RMS
                - f19, f20, f21, f22, f23, f24, f25, f26, f27, f28
            
            x : (segments, time, features)
    
            Returns (segments, 11, features)
        """
        y = np.abs(rfft(x, axis=1)) # amplitude of frequencies
        freq = np.expand_dims(rfftfreq(x.shape[1], sampling_freq), (0,-1))
        freq = np.repeat(freq, x.shape[0], axis=0)
        freq = np.repeat(freq, x.shape[2], axis=-1)

        # Mean, Std
        f12 = np.mean(y, axis=1, keepdims=True)
        f13 = np.std(y, axis=1, keepdims=True)
    
        # Skewness
        f14 = np.mean( ((y - f12)/f13)**3, axis=1, keepdims=True)
    
         # Kurtosis
        f15 = np.mean( ((y - f12)/f13)**4, axis=1, keepdims=True)
    
        f16 = np.sum(y * freq, axis=1, keepdims=True) / np.sum(y, axis=1, keepdims=True)
    
        f17 = np.mean(y*(freq - f16)**2, axis=1, keepdims=True)**0.5
        
        # RMS
        f18 = (np.sum(y*freq**2, axis=1, keepdims=True) / np.sum(y, axis=1, keepdims=True))**0.5
    
        # Others
        f19 = (np.sum(y*freq**4, axis=1, keepdims=True) / np.sum(y*freq**2, axis=1, keepdims=True))**0.5
    
        f20 = np.sum(y*freq**2, axis=1, keepdims=True) / (np.sum(y*freq**4, axis=1, keepdims=True) * np.sum(y, axis=1, keepdims=True))**0.5
    
        f21 = f17/f16
    
        f22 = np.mean(y * ((freq - f16)/f17)**3, axis=1, keepdims=True)
    
        f23 = np.mean(y * ((freq - f16)/f17)**4, axis=1, keepdims=True)
    
        f24 = np.mean(y * np.abs(freq - f16)/f17**0.5, axis=1, keepdims=True)
    
        f25 = np.mean(freq**2, axis=1, keepdims=True)**0.5
    
        f26 = np.sum(y * freq**0.5, axis=1, keepdims=True)
    
        f27 = 1 - np.sum(y * np.abs(freq - f26), axis=1, keepdims=True) / np.sum(y*freq, axis=1, keepdims=True)
    
        f28 = np.mean(((y - f12)**4) / f13, axis=1, keepdims=True)
    
        x_new = np.hstack([f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28])
    
        return x_new

    def monotonicty(self, 
                    x: np.array
                    )->float:
        """
            Computes the monotonicity metric coefficient of an array.

            Parameters
            ----------
            x : np.array
                The data array.

            Returns
            ----------
            m : float
                The monotonicty coefficient.
        """
        x_old = x[:-1]
        x_new = x[1:]
        return np.mean((x_new > x_old) + (-1)*(x_new < x_old))

    def feature_indicators(self, 
                           x: np.array, 
                           rul: np.array,
                           lengths : np.array,
                           )->np.array:
        """
            Computes the scores (feature indicators) of each feature averaged over every unit.
            Scores include :
                - Pearson correlation coefficient w.r.t. rul.
                - Spearman's rank correlation coefficient w.r.t. rul.
                - Monotonic metric coefficient

            Parameters
            ----------
            x : np.array
                The training data array of shape (segments, features).
            
            rul : np.array
                The associated rul labels array of shape (segments, 1).

            lengths : np.array
                The lengths of each unit.

            Returns
            ----------
            scores : np.array
                The feature indicators array of shape (features, metrics).
        """
        nb_features = x.shape[1]
        nb_units = len(lengths)
        scores = np.empty((nb_features, 3))
    
        for i in range(nb_features):
            f = x[:,i]
            l = rul[:,0]

            units_metrics = []
            for u in range(nb_units):
                li = [0]+list(np.cumsum(lengths))
                a, b = li[u], li[u+1] # at this point, samples are segments of each unit
                sub_f = f[a:b]
                sub_l = l[a:b]

                units_metrics.append(np.array([pearsonr(sub_f, sub_l).statistic, 
                                              spearmanr(sub_f, sub_l).statistic, 
                                              self.monotonicty(sub_f)]))
            scores[i] = np.nanmean(units_metrics, axis=0)
        return scores

    def get_windows_and_labels(self, ds, keep_last=False):
        # each row is a nb_features vector corresponding to 1 segment
        X, y, lengths = [], [], []
        unit_list = ds['unit'].unique()
        
        for unit in unit_list:
            x = ds[ds['unit']==unit].copy()
    
            # Overlapping window slicing
            if keep_last:
                windows = [x.drop(columns=["unit","RUL"]).to_numpy()[-self._WINDOW:]]
                labels = x['RUL'].iloc[-1:]
                
            else:
                windows = [x.drop(columns=["unit","RUL"]).to_numpy()[k:k+self._WINDOW] for k in range(len(x)-self._WINDOW+1)] # converts the time series of 1 sample into a list of windows
                labels = x['RUL'].iloc[self._WINDOW-1:] # related labels (1 label = window's last cycle's RUL value)
    
            # Adding samples, labels and lengths
            X += windows
            y = np.concatenate([y, labels]) # labels related to each window
            lengths.append(len(x)) # lengths of original time series
        
        # Final formatting
        X = np.array(X)
        y = np.expand_dims(y,-1)
        
        return X, y, lengths
            
    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        self._models = [GCUTCN() for k in range(self._FOLDS)]
        optimizers = [tf.keras.optimizers.Adam(learning_rate=self._LEARNING_RATE) for k in range(self._FOLDS)]
        
        for k in range(self._FOLDS):
            if self._train_datasets is not None:
                _ = self._models[k](np.random.randn(2, self._WINDOW, self._NB_FEATURES))
            elif self._test_dataset is not None:
                _ = self._models[k](np.random.randn(2, self._WINDOW, self._NB_FEATURES))
            else:
                raise AttributeError("Model has no defined train or test datasets to use.")

            self._models[k].compile(optimizer=optimizers[k], loss=tf.keras.losses.MeanSquaredError)


    def fit_model(self, 
                  verbose: bool=True)->None:
        """
            Fits each model to their respective training fold.

            Parameters
            ----------
            verbose : bool
                Whether to display fitting information.
        """
        if self._models is None:
            raise Exception("No models found. Call model.build_model() before.")
            
        self._hist = []
        for k in tqdm(range(self._FOLDS), desc="fitting models"):
            X_train, y_train = self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k]
            X_val, y_val = self._val_datasets["fold_data"][k], self._val_datasets["fold_labels"][k]

            NB_BATCH = len(X_train) // self._BATCH_SIZE
            """ocs = OneCycleScheduler(lr_max=self._LEARNING_RATE, 
                                    steps=self._EPOCHS*NB_BATCH, 
                                    mom_min=self._ONECLR_MOM_MIN,
                                    mom_max=self._ONECLR_MOM_MAX, 
                                    phase_1_pct=self._ONECLR_PHASE1_PCT, 
                                    div_factor=self._ONECLR_DIV_FACTOR)"""
            hist = self._models[k].fit(x=X_train, y=y_train,
                                      validation_data=(X_val, y_val),
                                      epochs=self._EPOCHS,
                                      batch_size=self._BATCH_SIZE,
                                      shuffle=True,
                                      verbose=verbose)
                                      #callbacks=[ocs])
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
            lr_exp = trial.suggest_int("learning_rate_exp", 1, 8)
            self._LEARNING_RATE = lr_factor * 10**(-lr_exp)
            if self._dataset_name == "femto":
                max_epochs = 300
            else:
                max_epochs = 100
            self._EPOCHS = trial.suggest_int("epochs", 10, max_epochs, step=10)
        
            set_seed(self._SEED) # doesn't affect sampler's state

            dataloader.build_train_val_data(self, data_path=data_load_path)

            self.fit_model(verbose=0)
            
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
        hp_names = ["LEARNING_RATE", "EPOCHS"]
        hp_vals = [self._LEARNING_RATE, self._EPOCHS]
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
                                          "NB_FEATURES",
                                          "WINDOW",
                                          "EPOCHS",
                                          "BATCH_SIZE",
                                          "LEARNING_RATE",
                                          "ONECLR_MOM_MIN",
                                          "ONECLR_MOM_MAX",
                                          "ONECLR_PHASE1_PCT",
                                          "ONECLR_DIV_FACTOR",
                                          "SEGMENT_LEN",
                                          "SEED"], 
              "value":[self._dataset_name,
                       self._FOLDS,
                       self._VARS_TO_KEEP,
                       self._NB_FEATURES,
                       self._WINDOW,
                       self._EPOCHS,
                       self._BATCH_SIZE,
                       self._LEARNING_RATE,
                       self._ONECLR_MOM_MIN,
                       self._ONECLR_MOM_MAX,
                       self._ONECLR_PHASE1_PCT,
                       self._ONECLR_DIV_FACTOR,
                       self._SEGMENT_LEN,
                       self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          False,
                          True,
                          False,
                          True,
                          False,
                          False,
                          False,
                          False,
                          False,
                          False]})
        res.to_csv(path+"_config.csv")
    