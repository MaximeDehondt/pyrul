# file_management.py
# Deals with directories and files used for plotting or saving hyperparameter searches.

import os
import glob
import pickle
import optuna

def clear_tmp_plots(path: str)->None:
    """
        Clears the given directory of all .png files.
        
        Parameters
        ----------
        path : str
            The path of the directory to clear.
    """
    L_files_to_remove = glob.glob(os.path.join(path, "*.png"))
    for filePath in L_files_to_remove:
        os.remove(filePath)

class SamplerStateSaveCallback:
    """
        Saves the state of the hyperparameter sampler when called.
    """
    def __init__(self)->None:
        pass

    def __call__(self, 
                 study: optuna.study.Study, 
                 trial: optuna.trial.FrozenTrial
                 )->None:
        with open("sampler.pkl", "wb") as fout:
            pickle.dump(study.sampler, fout)

class SaveBestDataOnly():
    """
        Saves the models, training and validation datasets, along with the data metrics, of the current best model.
        Previous best model is overwritten. 
        During kernel crashes, this function can be called again as long as best_trial_idx is correct.
    """
    def __init__(self, 
                 dl,
                 model,
                 data_path: str, 
                 data_metrics_path: str, 
                 model_path: str,
                 best_trial_idx: int=-1
                 )->None:
        """
            Parameters
            ----------
            dl : DataLoader
                The dataloader object associated to the model.

            model : RULModel
                The prediction model.

            data_path : str
                The path to save the data.

            data_metrics_path : str
                The path to save the data metrics.

            model_path : str
                The path to save the models.

            best_trial_idx : int, default=-1
                The initial best trial index.
        """

        self._dl = dl
        self._model = model
        self._data_path = data_path
        self._data_metrics_path = data_metrics_path
        self._model_path = model_path
        self._best_idx = best_trial_idx

    def __call__(self, 
                 study: optuna.study.Study, 
                 trial: optuna.trial.FrozenTrial
                 )->None:
        
        if self._best_idx != study.best_trial.number: # if better model found
            
            # Saving best model and its info
            self._best_idx = study.best_trial.number
            self._dl.save_datasets(self._model, data=["train", "val"], path=self._data_path)
            self._dl.save_data_metrics(self._model, path=self._data_metrics_path)
            self._model.save_models(path=self._model_path)
            self._model.save_hp_values(path=self._model_path)