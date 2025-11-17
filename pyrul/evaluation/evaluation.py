# evaluation.py
# To assess the performances of the RUL prediction models.

import numpy as np
import pandas as pd
from .metrics import *
from pyrul.models.base import RULModel 
from pyrul.utils.various import tensor_to_numpy

class Evaluator():
    """
        A class to get the performances of the RUL prediction models.
    """
    def __init__(self, metrics: list):
        """
            Parameters
            ----------
            metrics : list of str
                The names of all of the metrics to consider. 
                It is possible to specify if thresholded or non-thresholed data should be looked at.
                The phase to study can be specified before the metric name with "phase0_" for thresholded data, and "phase1_" for non-thresholded data.
                Eg: "phase1_rmse" to evaluate the root mean squared error on non-thresholded data.
        """
        # Note : New metrics should be referenced here.
        dict_fun = dict({"rmse": Metric_RMSE, 
                         "mae": Metric_MAE, 
                         "scoring_cmapss": Metric_Scoring_CMAPSS, 
                         "scoring_femto": Metric_Scoring_FEMTO,
                         "mape": Metric_MAPE,
                         "pmae": Metric_PMAE,
                         "cv": Metric_CV,
                         "r_abs_err": Metric_RAE,
                         "it": Metric_IT,
                         "sstd": Metric_SSTD,
                         "rt95": Metric_RT95,
                         "pearsonr": Metric_PEARSON})

        # Instantiation of all desired metrics
        self._metrics = dict({})
        for m in metrics:
            
            if (m[:7] == "phase0_"):
                m_name = m[7:]
                phase = 0
            elif (m[:7] == "phase1_"):
                m_name = m[7:]
                phase = 1
            else: 
                m_name = m
                phase= -1
                
            if m_name in dict_fun.keys():
                self._metrics[m] = dict_fun[m_name](phase=phase)
            else:
                raise ValueError("Metric name '",m,"'not found.")

    def get_core_metrics(self, 
                    X: np.array, 
                    y: np.array, 
                    lengths: list,
                    data_type : str,
                    model: RULModel, 
                    k: int,
                    save_pred_path: str=None,
                    load_pred_path: str=None
                    )->list:
        """
            Returns a dataframe made of metric values for the k-th model, for the given data.
            If k=-1, uses the average prediction of all models.

            Parameters
            ----------
            X : array-like
                The data features.

            y : array-like
                The data labels.

            lengths : array-like
                The length of each unit.

            data_type : str
                The type of the data. Is either "train", "val", "test" or "test_full".

            model : RULModel
                The prediction model.

            k : int
                The index of the model to choose, can be -1 for the average prediction of all models.

            save_pred_path : str, default=None
                The path to save model predictions to avoid future identical computations. If None, no saving is done.

            load_pred_path : str, default=None
                The path to load model predictions to avoid inference computations. If None, no loading is done.

            Returns
            ----------
            res : list
                Metric values of the desired model for the given data.
                
        """
        # if model._is_ibl and data_type=="test_full": # To avoid unecessary calls to predict() for IBL models since test_full is same as test for them
        #     return [np.nan for _ in self._metrics.values()]

        # Step 1. Checking whether to compute or load predictions
        if load_pred_path is not None:
            y_pred = np.loadtxt(fname=load_pred_path+"_pred.txt", delimiter=",")
 
        else:
            if model._is_ibl and data_type == "test_full": # Full test prediction for IBL models, on test data
                y_pred = model.predict(X, -1, is_test=True, do_full_pred=True)
                
            else: # Regular model.predict() for other models
                is_test = data_type in ["test", "test_full"]
                y_pred = model.predict(X, k, is_test=is_test)
                y_pred = tensor_to_numpy(y_pred)
                if model._is_deep_lstm:
                    y_pred = y_pred[:,:,0]
            
        if save_pred_path is not None:
            np.savetxt(X=y_pred, fname=save_pred_path+"_pred.txt", delimiter=",")

        # Step 2. Formatting y and y_pred depending on the model
        if model._is_ibl:
            if data_type == "test_full":
                y = np.concatenate([traj[model._MIN_TS_SIZE-1:] for traj in y])
            else:
                y = np.array([traj[-1] for traj in y])
                
        elif model._is_deep_lstm:
            y = y[:,:,0]
            #y_pred = y_pred[:,:,0]

            if data_type=="test":
                y = np.array([y[i][lengths[i]-1] for i in range(len(lengths))])
                y_pred = np.array([y_pred[i][lengths[i]-1] for i in range(len(lengths))])

            else:
                mask = y!=-1.
                y, y_pred = y[mask], y_pred[mask] # masking flattens array

            y = tensor_to_numpy(y)
            y_pred = tensor_to_numpy(y_pred)
            
        else:
            y = tensor_to_numpy(y)
            if y.ndim==2:
                y = y[:,0]
            if y_pred.ndim==2:
                y_pred = y_pred[:,0]

        #if model._is_rul_normalized: # UNUSED
        #    y = model.denorm(y, data_type, k, lengths)
        #    y_pred = model.denorm(y_pred, data_type, k, lengths)

        # Step 3. Calling function evaluation for each metric
        res = [fun(y_true=y, 
                   y_pred=y_pred,
                   lengths=lengths, 
                   data_type=data_type,
                   model=model,
                   k=k) 
               for fun in self._metrics.values()]
        
        return res

    def get_metrics(self, 
                    model: RULModel, 
                    data: str,
                    save_pred_path: str=None,
                    load_pred_path: str=None
                    )->pd.DataFrame:
        """
            Returns a dataframe made of metric values for each model, for the desired data.

            Parameters
            ----------
            model : RULModel
                The prediction model.

            data : str
                The list of data types to consider. Should be either "train", "val", "test" or "test_full".
                Can also be "test_ind" and "test_full_ind", to assess model performance per fold on test or test_full data, instead of only averaging model predictions.

            save_pred_path : str, default=None
                The path to save model predictions to avoid future identical computations. If None, no saving is done.

            load_pred_path : str, default=None
                The path to load model predictions to avoid inference computations. If None, no loading is done.

            Returns
            ----------
            df : pd.DataFrame
                Metric values for each model and for the desired data.
        """
        n = len(self._metrics.keys())
        empty_lists = [[] for _ in range(n)]
        
        list_keys = [data+"_"+k for k in self._metrics.keys()]
        metrics = dict(zip(list_keys, empty_lists)) # eg: dict({"train_rmse": [], "train_mae":[], ...,})

        if data == "train":
            ds = model._train_datasets
        elif data == "val":
            ds = model._val_datasets
        elif data in ["test","test_ind"]:
            ds = model._test_dataset
        elif data in ["test_full", "test_full_ind"]:
            ds = model._test_full_dataset
        else:
            raise ValueError("Data type not found. Choose between 'train', 'val', 'test' or 'test_full'.")

        if data in ["train","val"]:
            for k in range(model._FOLDS):
                s_path = save_pred_path+data+"_fold_"+str(k) if save_pred_path is not None else None
                l_path = load_pred_path+data+"_fold_"+str(k) if load_pred_path is not None else None
                res = self.get_core_metrics(X=ds["fold_data"][k], 
                                            y=ds["fold_labels"][k],
                                            lengths=ds["fold_lengths"][k],
                                            data_type=data,
                                            model=model,
                                            k=k,
                                            save_pred_path=s_path,
                                            load_pred_path=l_path)
                
                for i, key in enumerate(list_keys):
                    metrics[key].append(res[i])
        else:
            if data in ["test_ind","test_full_ind"]:
                for k in range(model._FOLDS):
                    s_path = save_pred_path+data+"_fold_"+str(k) if save_pred_path is not None else None
                    l_path = load_pred_path+data+"_fold_"+str(k) if load_pred_path is not None else None
                    true_data_name = ("test" if data=="test_ind" else "test_full")
                    res = self.get_core_metrics(X=ds["data"], 
                                                y=ds["labels"],
                                                lengths=ds["lengths"],
                                                data_type=true_data_name,
                                                model=model,
                                                k=k,
                                                save_pred_path=s_path,
                                                load_pred_path=l_path)
                    
                    for i, key in enumerate(list_keys):
                        metrics[key].append(res[i])
                    
            else:
                s_path = save_pred_path+data if save_pred_path is not None else None
                l_path = load_pred_path+data if load_pred_path is not None else None
            
                res = self.get_core_metrics(X=ds["data"], 
                                            y=ds["labels"],
                                            lengths=ds["lengths"],
                                            data_type=data,
                                            model=model,
                                            k=-1,
                                            save_pred_path=s_path,
                                            load_pred_path=l_path)

                for i, key in enumerate(list_keys):
                    metrics[key].append(res[i])
    
        return pd.DataFrame(metrics)
            

    def get_unit_pred(self,
                      model: RULModel,
                      data,
                      unit,
                      lengths,
                      k: int,
                      threshold=None,
                      ignore_threshold=None
                      )->np.array:
        """
            Returns the prediction of the chosen model on the given data.

            Parameters
            ----------
            model : RULModel
                The RUL prediction model.

            data : array-like
                The feature data.

            unit : int
                The index of the unit.

            lengths : list of int
                The lengths of every unit.

            k : int
                The model to choose.

            Returns
            ----------
            y_pred : np.array
                The RUL prediction of the unit.
        """
        slice_inf, slice_sup = model.get_slices(unit, lengths)
        X = data[slice_inf:slice_sup]
        y_pred = tensor_to_numpy(model.predict(data, k))
        return y_pred