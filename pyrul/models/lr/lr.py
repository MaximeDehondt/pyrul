# lr.py

from tqdm import tqdm
import numpy as np
import pandas as pd
from pyrul.models.base import RULModel
from sklearn.linear_model import LinearRegression as LR
import pickle
from pyrul.utils.seed import set_seed
from pyrul.utils.dataset_info import get_defaults_vars
from pyrul.evaluation.metrics import rmse

class LR_Model(RULModel):
    """
        Implementation of a simple linear regressor baseline.
    """

    def __init__(self,
                 DATASET_NAME: str,
                 FOLDS: int = None,
                 VARS_TO_KEEP: list = None,
                 WINDOW: int = 30,
                 R_EARLY: float = None,
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

            WINDOW : int
                The size of the window for the input data.

            R_EARLY : int, default=None
                The RUL threshold to apply to the train and validation labels. If None, no threshold is applied.

            SEED: int, default=None
                The random seed for reproducible results.
        """
        super().__init__(dataset_name=DATASET_NAME, 
                         dataset_ori="cmapss_fd001",
                         folds=FOLDS, 
                         seed=SEED)

        self._VARS_TO_KEEP = get_defaults_vars(DATASET_NAME, "cmapss_fd001") if VARS_TO_KEEP is None else VARS_TO_KEEP
        self._WINDOW = WINDOW
        self._R_EARLY = R_EARLY

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
        
        # Z-norm     
        self._df_train_means[k] = df[self._VARS_TO_KEEP].mean()
        self._df_train_stds[k] = df[self._VARS_TO_KEEP].std()
        
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
            
        keep_last = data_type == "test"
        apply_thres = data_type in ["train","val"]

        # Step 1. Selecting initial features
        df = df[["unit","RUL"]+self._VARS_TO_KEEP]

        # Step 2. RUL threshold
        if apply_thres and self._R_EARLY is not None:
            df.loc[:,"RUL"] = np.minimum(self._R_EARLY, df["RUL"])

        # Step 3. Feature selection and Z-norm scaling (except for "unit" and "RUL")
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

        # Step 4. Individual pre-processing
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
        return self.data_process_global_transform(datasets=train_datasets, data_type="train")

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
            
    ###############################################################
    ##################### MODEL CREATION ##########################
    ###############################################################

    def build_model(self)->None:
        """
            Instanciates a model for each fold.
        """
        self._models = [LR() for k in range(self._FOLDS)]

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
        # This is a generic example of how to use hp search.

        def hp_trial(trial):

            # Hyperparameters to search
            self._WINDOW = trial.suggest_int("window", 10, 30, step=5)
            
            set_seed(self._SEED) # doesn't affect sampler's state

            # Data processing
            dataloader.build_train_val_data(self, data_path=data_load_path)

            # Model building and fitting
            self.build_model()
            self.fit_model()

            # Model evaluation on validation data
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
                The path to load the models.
        """            
        self._models = []
        for k in range(self._FOLDS):
            with open(path+'_model_'+str(k)+'.pkl', 'rb') as f:
                self._models.append(pickle.load(f))

    ###############################################################
    ##################### UTILITIES ###############################
    ###############################################################

    def get_slices(self, 
                   idx: int, 
                   lengths: list,
                   data_type: str
                   )->tuple:
        """
            Returns 2 indices to select the idx-th unit.
            Beware that due to data preprocess, the indices of time series do not correspond to the original units'.

            Parameters
            ----------  
            idx : int
                The index of the fold's original time series. 

            lengths : list
                Lengths of each unit.

            data_type : str
                Is either "train", "val", "test" or "test_full".

            Returns
            ----------
            slice_inf, slice_sup : tuple
                The indices to select the desired unit.
        """
        if data_type not in ["train","val","test","test_full"]:
            raise ValueError("Data type '"+data_type+"' not found.")
            
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
                Whether the trajectory is in the full test dataset or not.

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
            Saves the current hyperparameter values in a csv file.

            Parameters
            ----------
            path : str
                The path to save the hyperparameter values.
        """
        hp_names = ["WINDOW"]
        hp_vals = [self._WINDOW]
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
                                          "R_EARLY",
                                          "SEED"], 
              "value":[self._dataset_name,
                      self._FOLDS,
                      self._VARS_TO_KEEP,
                      self._WINDOW,
                      self._R_EARLY,
                      self._SEED], 
              "is_tuned":[False,
                         False,
                         False,
                         False,
                         False,
                         False]})
        res.to_csv(path+"_config.csv")