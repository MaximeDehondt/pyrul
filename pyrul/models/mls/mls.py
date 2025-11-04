# mls.py
from tqdm import tqdm
import numpy as np
import pandas as pd
from .mls_core import MLS
from pyrul.models.base import RULModel
from pyrul.utils.seed import set_seed
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from pyrul.utils.dataset_info import get_defaults_vars, get_min_ts_size

class MLS_Model(RULModel):
    """
        Reimplementation of the prediction model in the paper
        "Remaining useful life estimation with multiple local similarities" (Jianhua Lyu, 2020)
    """

    def __init__(self,
                 DATASET_NAME,
                 FOLDS = None,
                 VARS_TO_KEEP = None,
                 R_EARLY = 130,
                 MIN_TS_SIZE = None,
                 WINDOW = 30,
                 SLIDING_STEP = 15,
                 GF_SIGMA = 5,
                 ALPHA = 0,
                 BETA = 0,
                 LAMBDA1 = 0,
                 LAMBDA2 = 0,
                 DELTA_W = 2,
                 K = 4,
                 KERNEL_BANDWIDTH = 1,
                 SEED=None):
        """
            DATASET_NAME : str
                The name of the dataset to use.

            FOLDS : int
                The number of folds to create for the k-fold cross-validation split. If None, default values will apply depending on DATASET_NAME.

            VARS_TO_KEEP : list of str, default=None
                The list of variables to keep. If None, default values will apply depending on DATASET_NAME.

            R_EARLY : int, default=130
                The RUL threshold to apply to the train and validation labels. If None, no threshold is applied.

            MIN_TS_SIZE : int, default=None
                The minimum size of a time series. If None, default values will apply depending on DATASET_NAME.

            WINDOW : int, default=30
                The window length.

            SLIDING_STEP : int, default=15
                The sliding window step.

            GF_SIGMA : float
                The Gaussian filter gamma parameter.

            ALPHA : float
                The first adjusting function parameter.

            BETA : float
                The second adjusting function parameter.

            LAMBDA1 :

            LAMBDA2 :

            DELTA_W : int
                The maximum offset allowed in the window comparison process.

            K : int
                The number of DTW alternate comparisons allowed.

            KERNEL_BANDWIDTH : float, default=1.
                The kernel bandwidth used to compute the predicted RUL density function.
                Does not influence the predicted rul values.

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
        self._WINDOW = WINDOW
        self._SLIDING_STEP = SLIDING_STEP
        self._GF_SIGMA = GF_SIGMA
        self._ALPHA = ALPHA
        self._BETA = BETA
        self._LAMBDA1 = LAMBDA1
        self._LAMBDA2 = LAMBDA2
        self._DELTA_W = DELTA_W
        self._K = K
        self._KERNEL_BANDWIDTH = KERNEL_BANDWIDTH
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

        df = (df - df_train_min) / (df_train_max - df_train_min)
        df[["unit","RUL"]] = df_info
    
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
            Returns the array of samples, labels and original length, such that each sample is a time series of a unit.
    
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

            # Gaussian filter for each feature
            x_filt = gaussian_filter1d(x, sigma=self._GF_SIGMA, axis=0)
            
            # DI Construction (PCA)
            x_compressed = PCA(n_components=1).fit_transform(x_filt)[:,0]
    
            X.append(x_compressed)
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
        self._models = [MLS(window=self._WINDOW,
                            sliding_step=self._SLIDING_STEP,
                            k=self._K,
                            delta_w=self._DELTA_W,
                            alpha=self._ALPHA,
                            beta=self._BETA,
                            lambda1=self._LAMBDA1,
                            lambda2=self._LAMBDA2,
                            h=self._KERNEL_BANDWIDTH,
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
        
            self._GF_SIGMA = np.around(trial.suggest_float("gf_sigma", 0.01, 5., step=0.01), 2)
            self._ALPHA =  trial.suggest_int("alpha", 0, 1000)
            self._BETA =  trial.suggest_int("beta", 0, 1000)
            self._LAMBDA1 =  trial.suggest_int("lambda1", 0, 1000)
            self._LAMBDA2 = trial.suggest_int("lambda2", 0, 1000)
            self._DELTA_W = trial.suggest_int("delta_w", 0, 5)
            if self._dataset_name == "cmapss_fd001":
                self._WINDOW = trial.suggest_int("window", 10, 30, step=10)
            elif self._dataset_name == "femto":
                self._WINDOW = trial.suggest_int("window", 10, 80, step=10)

            self._SLIDING_STEP = self._WINDOW//2
            self._K = trial.suggest_int("k", 1, 4)
        
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
        hp_names = ["GF_SIGMA", "ALPHA", "BETA", "LAMBDA1", "LAMBDA2", "DELTA_W", "K", "WINDOW"]
        hp_vals = [self._GF_SIGMA, self._ALPHA, self._BETA, self._LAMBDA1, self._LAMBDA2, self._DELTA_W, self._K, self._WINDOW]
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
                                          "WINDOW",
                                          "SLIDING_STEP",
                                          "GF_SIGMA",
                                          "ALPHA",
                                          "BETA",
                                          "LAMBDA1",
                                          "LAMBDA2",
                                          "DELTA_W",
                                          "K",
                                          "KERNEL_BANDWIDTH",
                                          "SEED"], 
              "value":[self._dataset_name,
                       self._FOLDS,
                       self._VARS_TO_KEEP,
                       self._R_EARLY,
                       self._MIN_TS_SIZE,
                       self._WINDOW,
                       self._SLIDING_STEP,
                       self._GF_SIGMA,
                       self._ALPHA,
                       self._BETA,
                       self._LAMBDA1,
                       self._LAMBDA2,
                       self._DELTA_W,
                       self._K,
                       self._KERNEL_BANDWIDTH,
                       self._SEED], 
              "is_tuned":[False,
                          False,
                          False,
                          False,
                          False,
                          True,
                          False,
                          True,
                          True,
                          True,
                          True,
                          True,
                          True,
                          True,
                          False,
                          False]})
        res.to_csv(path+"_config.csv")