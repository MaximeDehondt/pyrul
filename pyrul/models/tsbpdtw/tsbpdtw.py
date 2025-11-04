# tsbpdtw.py
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from .tsbpdtw_core import TSBP
from pyrul.models.base import RULModel
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars, get_min_ts_size

class TSBPDTW_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Trajectory Similarity Matching and Remaining Useful Life Prediction Based on Dynamic Time Warping" (Lin Huang, 2022)
    """

    def __init__(self,
                 DATASET_NAME,
                 FOLDS = None,
                 VARS_TO_KEEP = None,
                 R_EARLY = 125,
                 MIN_TS_SIZE = None,
                 KR_ALPHA = 0.1, 
                 KR_GAMMA = 0.01,
                 DTW_TOL = 4,
                 NB_SIM = 7,
                 INTER_FACTOR = 0.385,
                 D1 = 145,
                 D2 = 125,
                 SEED = None):
        """
            FOLDS : int
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depending on DATASET_NAME.

            R_EARLY : int, default=None
                The RUL threshold to apply to the train and validation labels. If None, no threshold is applied.

            MIN_TS_SIZE: int, default=None
                The minimum trajectory size to build validation data. If None, default values will apply depending on DATASET_NAME.

            KR_ALPHA: float, default=0.1
                Kernel regression alpha parameter.

            KR_GAMMA: float, default=0.01
                Kernel regression gamma parameter.

            DTW_TOL: int, default=4
                The tolerance value of the DTW search.

            NB_SIM: int, default=7
                The number of similar trajectories to use for RUL prediction.

            INTER_FACTOR: float, default=0.385
                Proportion of the weight assigned to the smallest similar trajectory's RUL value, w.r.t. the one assigned to the highest similar trajectory's RUL value.

            D1: int, default=145
                Upper reference length discrepancy threshold. Reference trajectories with a length discrepancy >= D1 will be ignored.

            D2: int, default=125
                Lower reference length discrepancy threshold. Reference trajectories with a length discrepancy < D2 will be ignored.

            SEED: int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME, 
                         dataset_ori="cmapss_fd001",
                         folds=FOLDS, 
                         is_ibl=True,
                         seed=SEED)

        self._R_EARLY = R_EARLY
        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._MIN_TS_SIZE = get_min_ts_size(DATASET_NAME) if MIN_TS_SIZE is None else MIN_TS_SIZE
        self._KR_ALPHA = KR_ALPHA
        self._KR_GAMMA = KR_GAMMA
        self._DTW_TOL = DTW_TOL
        self._NB_SIM = NB_SIM
        self._INTER_FACTOR = INTER_FACTOR
        self._D1 = D1
        self._D2 = D2
        self._TIME_FACTOR = (10 if DATASET_NAME=="femto" else 1)

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
        rd_trunc = data_type == "val"
        apply_thres = data_type in ["train","val"]
        
        if rd_trunc:
            # Converting validation data into incomplete-run-to-failure
            df = df.groupby('unit', sort=False).apply(lambda x: x.iloc[:np.random.randint(self._MIN_TS_SIZE, len(x)+1)]).reset_index(drop=True)
        
        # Step 1. Piecewise Linear RUL labels
        if apply_thres and self._R_EARLY is not None:
            df['RUL'] = np.minimum(self._R_EARLY, df['RUL'])
    
        # Step 2. Feature selection and Max-norm scaling (except for "engine" and "RUL")
        df_old_vars = df[["unit","RUL"]].copy()
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

        df = (df - df_train_min) / (df_train_max - df_train_min)
        df[["unit","RUL"]] = df_old_vars
    
        # Step 3. Individual pre-processing
        X, y, lengths = self.transform_units(df)
    
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
        self._df_train_min.to_csv(path+"_df_train_min.csv", sep=",")
        self._df_train_max.to_csv(path+"_df_train_max.csv", sep=",")

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

    def transform_units(self, 
                        ds: pd.DataFrame
                        )->tuple:
        """
            Returns the array of samples, labels and original length, such that each sample is a unit time series.
    
            Parameters
            ----------
            ds : pd.DataFrame
                The dataset. Must have "unit" and "RUL" as columns at least.
                
            Returns
            ----------
            X : np.array
                The 1D feature values after PCA and smoothing.
            
            y : np.array
                The related RUL values.
            
            lengths : np.array
                The lengths of the original unit.
        """
        X, y, lengths = [], [], []
        unit_list = ds['unit'].unique()
    
        for unit in ds['unit'].unique():
            x = ds[ds['unit']==unit]
            rul = x['RUL'].values
            x = x[self._VARS_TO_KEEP].values
            
            # DI Construction (PCA)
            x_compressed = PCA(n_components=1).fit_transform(x)[:,0]
    
            # Kernel Ridge smoothing
            kr = KernelRidge(alpha=self._KR_ALPHA, kernel='rbf', gamma=self._KR_GAMMA)
            kr = kr.fit(np.expand_dims(rul,-1), x_compressed)
            x_smoothed = kr.predict(np.expand_dims(rul,-1))
        
            X.append(x_smoothed)
            y.append(rul)
            lengths.append(len(x))
    
        return X, y, lengths

    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        self._models = [TSBP(dtw_tol=self._DTW_TOL, 
                               nb_sim=self._NB_SIM, 
                               k=self._INTER_FACTOR,
                               d1=self._D1,
                               d2=self._D2,
                               r_early=self._R_EARLY,
                               time_factor=self._TIME_FACTOR) for k in range(self._FOLDS)]

    def fit_model(self)->None:
        """
            Fits each model to their respective training fold.
        """
        for k in tqdm(range(self._FOLDS), desc="fitting models"):
            self._models[k].fit(self._train_datasets["fold_data"][k], self._train_datasets["fold_labels"][k])

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
        
            self._KR_ALPHA = np.around(trial.suggest_float("kr_alpha", 0.01, 10), 2)
            self._KR_GAMMA = np.around(trial.suggest_float("kr_gamma", 0.01, 10), 2)
            self._DTW_TOL = trial.suggest_int("dtw_tol", 0, 10)
            
            if self._dataset_name=="femto":
                self._NB_SIM = trial.suggest_int("nb_sim", 1, 5)
            else:
                self._NB_SIM = trial.suggest_int("nb_sim", 1, 10)
                
            self._INTER_FACTOR = np.around(trial.suggest_float("inter_factor", 0, 1), 3)

            if self._dataset_name=="femto":
                self._D1 = trial.suggest_int("d1", 1, 3000)
                self._D2 = trial.suggest_int("d2", 1, 3000)
            else:
                self._D1 = trial.suggest_int("d1", 1, 300)
                self._D2 = trial.suggest_int("d2", 1, 300)
        
            set_seed(self._SEED)

            dataloader.build_train_val_data(self, data_path=data_load_path)
            
            self.build_model()
            self.fit_model()
            
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
        pass

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################

    def get_slices(self, 
                   idx: int, 
                   lengths: np.array,
                   data_type: str,
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
                Not used. Lengths of each unit.

            data_type : str
                Is either "train", "val", "test" or "test_full".

            Returns
            ----------
            slice_inf, slice_sup : tuple
                The indices to select the desired unit.
        """
        return idx, idx+1

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
        return np.arange(1, length+1) * (10 if self._dataset_name=="femto" else 1)


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
        hp_names = ["KR_ALPHA", "KR_GAMMA", "DTW_TOL", "NB_SIM", "INTER_FACTOR", "D1", "D2"]
        hp_vals = [self._KR_ALPHA, self._KR_GAMMA, self._DTW_TOL, self._NB_SIM, self._INTER_FACTOR, self._D1, self._D2]
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
                                          "MIN_TS_SIZE",
                                          "KR_ALPHA",
                                          "KR_GAMMA",
                                          "DTW_TOL",
                                          "NB_SIM",
                                          "INTER_FACTOR",
                                          "D1",
                                          "D2",
                                          "SEED"], 
              "value":[self._dataset_name,
                       self._FOLDS,
                       self._VARS_TO_KEEP,
                       self._R_EARLY,
                       self._MIN_TS_SIZE,
                       self._KR_ALPHA,
                       self._KR_GAMMA,
                       self._DTW_TOL,
                       self._NB_SIM,
                       self._INTER_FACTOR,
                       self._D1,
                       self._D2,
                       self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          False,
                          True,
                          True,
                          True,
                          True,
                          True,
                          True,
                          True,
                          False]})
        res.to_csv(path+"_config.csv")