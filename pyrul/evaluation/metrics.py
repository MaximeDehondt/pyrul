# metrics.py : 
# Metric classes for performance assessment.

import numpy as np
from scipy.stats import pearsonr
import time

########## Utility metric functions ##########
class Metric():
    def __init__(self,
                 func,
                 kind: str,
                 phase: int):

        """
            Generic metric class to handle model specificities.

            Parameters
            ----------
            func : function
                The metric computation function to call.

            kind : str
                The type of the metric. Is either "point-wise", "unit-wise" or "model-wise".

            phase : int
                The RUL phase to consider. Is either 0, 1 or -1.
        """

        self._metric = func
        self._kind = kind
        self._phase = phase
    
    def __call__(self, 
                 y_true, 
                 y_pred, 
                 lengths, 
                 data_type: str, 
                 model, 
                 k: int
                 )->float:
        """
            Generic call function of the metric class, handling model specificities to properly compute the metrics with the right data.

            Parameters
            ----------
            y_true : np.array
                The true RUL labels.

            y_pred : np.array
                The predicted RUL labels.

            lengths : np.array
                The length of the original units.

            data_type : str
                The type of the data. Is either "train", "val", "test", "test_full".

            model : RULModel
                The predictive model.

            k : int
                The associated fold.

            Returns
            -------
            res : float
                The metric value.
        """
        
        if self._kind=="point_wise": # filtering points possible before point wise operations

            ori_size = np.size(y_true) # original nb of labels before filtering, to get metrics with additive properties
            
            if hasattr(model, "_R_EARLY") and model._R_EARLY is not None: # Phase selection
                y_true, y_pred = select_phase(phase=self._phase, threshold=model._R_EARLY, y_true=y_true, y_pred=y_pred)
                
            return self._metric(y_true=y_true, y_pred=y_pred, ori_size=ori_size)                   

        elif self._kind=="unit_wise": # filtering points during unit wise operations
            if data_type=="test":
                return np.nan
            else:
                res = []
                for i in range(len(lengths)): # evaluating every unit
                    if model._is_deep_lstm:
                        a, b = model.get_slices(i, lengths, data_type, is_unmasked=True)
                    elif model._is_ibl:
                        a, b = a, b = model.get_slices(i, lengths, data_type, is_pred_rul_compressed=(data_type=="test_full"))
                    else:
                        a, b = model.get_slices(i, lengths, data_type)

                    # Unit selection
                    y_true_sub = y_true[a:b]
                    y_pred_sub = y_pred[a:b]

                    # Phase selection
                    if hasattr(model, "_R_EARLY") and model._R_EARLY is not None:
                        y_true_sub, y_pred_sub = select_phase(phase=self._phase, threshold=model._R_EARLY, y_true=y_true_sub, y_pred=y_pred_sub)

                    if len(y_true_sub)>1:
                        res.append(self._metric(y_true=y_true_sub, y_pred=y_pred_sub))
                    
                return np.sum(res) / len(res)

        elif self._kind=="model_wise":
            if data_type == "test":
                return self._metric(model, data_type)
            else:
                return np.nan
        else:
            raise ValueError("Metric kind '{k}' not found.".format(k=self._kind))


def select_phase(phase: int,
                 threshold: float,
                 y_true: np.array,
                 y_pred: np.array,
                 )->tuple:
    """
        Based on phase, selects the thresholded or non-thresholded values, of y_true and y_pred using the values of y_true and the given r_early threshold.
        If phase is -1, selects everything (no filter).

        Parameters
        ----------
        phase : int
            The phase to select. Is either 0 for thresholded values or 1 for non-thresholded values.

        threshold : float
            The RUL threshold.

        y_true : np.array
            The true RUL labels.

        y_pred : np.array
            The predicted RUL labels.
            
        Returns
        -------
        y_true : np.array
            The filtered true RUL labels.

        y_pred : np.array
            The filtered predicted RUL labels.
    """
    if (phase==0): # select thresholded labels if possible
        mask = y_true>=threshold
    elif (phase==1): # select non-thresholed labels if possible
        mask = y_true<threshold
    elif (phase==-1):
        return y_true, y_pred
    else:
        raise ValueError("Phase "+str(phase)+" not implemented.")

    return y_true[mask], y_pred[mask]


####### List of specific metric classes below #######
class Metric_Scoring_CMAPSS(Metric):
    def __init__(self, phase):
        super().__init__(func=scoring_cmapss,
                         kind="point_wise", 
                         phase=phase)

class Metric_Scoring_FEMTO(Metric):
    def __init__(self, phase):
        super().__init__(func=scoring_femto,
                         kind="point_wise", 
                         phase=phase)
        
class Metric_RMSE(Metric):
    def __init__(self, phase):
        super().__init__(func=rmse,
                         kind="point_wise", 
                         phase=phase)

class Metric_MAE(Metric):
    def __init__(self, phase):
        super().__init__(func=mae,
                         kind="point_wise", 
                         phase=phase)

class Metric_PMAE(Metric):
    def __init__(self, phase):
        super().__init__(func=positive_mae,
                         kind="point_wise", 
                         phase=phase)

class Metric_MAPE(Metric):
    def __init__(self, phase):
        super().__init__(func=mape,
                         kind="point_wise", 
                         phase=phase)

class Metric_RAE(Metric):
    def __init__(self, phase):
        super().__init__(func=relative_abs_err,
                         kind="point_wise", 
                         phase=phase)

class Metric_IT(Metric):
    def __init__(self, phase):
        super().__init__(func=inference_time,
                         kind="model_wise", 
                         phase=phase)
        
class Metric_CV(Metric):
    def __init__(self, phase):
        super().__init__(func=convergence,
                         kind="unit_wise", 
                         phase=phase)

class Metric_SSTD(Metric):
    def __init__(self, phase):
        super().__init__(func=sample_std,
                         kind="unit_wise", 
                         phase=phase)

class Metric_RT95(Metric):
    def __init__(self, phase):
        super().__init__(func=relative_time_at_95,
                         kind="unit_wise", 
                         phase=phase)

class Metric_PEARSON(Metric):
    def __init__(self, phase):
        super().__init__(func=pearson_corr,
                         kind="point_wise", 
                         phase=phase)

####### List of specific metric functions below #######
def scoring_cmapss(y_true, 
                   y_pred,
                   **kwargs
                   )->float:
    """
        Returns the scoring (CMAPSS version) between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.
    
        Returns
        ----------
        scoring : float
            The scoring function value between y_true and y_pred.

    """
    d = y_pred - y_true
    scores = (d<0)*np.exp(-d/13) + (d>=0)*np.exp(d/10) - 1
    return np.sum(scores)

def scoring_femto(y_true, 
                  y_pred,
                  **kwargs
                  )->float:
    """
        Returns the scoring (FEMTO version) between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.
    
        Returns
        ----------
        scoring : float
            The scoring function value between y_true and y_pred.

    """
    rerr = 100*(y_true - y_pred)/y_true
    scores = (rerr<=0)*np.exp(-np.log(0.5)*rerr/5) + (rerr>0)*np.exp(np.log(0.5)*rerr/20)
    return np.sum(scores) / np.size(y_true)

def rmse(y_true, 
         y_pred,
         ori_size=None
         )->float:
    """
        Returns the root mean squared error between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        ori_size : int, default=None
            Optional. The number of values in each arrays.

        Returns
        ----------
        rmse : float
            The RMSE between y_true and y_pred.
    """
    if ori_size:
        n = ori_size
    else:
        n = np.size(y_true)
        
    return np.sqrt(np.sum(np.square(y_true - y_pred))/n)

def mae(y_true, 
        y_pred,
        ori_size=None
        )->float:
    """
        Returns the mean absolute error between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        ori_size : int, default=None
            Optional. The number of values in each arrays.

        Returns
        ----------
        mae : float
            The RMSE between y_true and y_pred.
    """ 
    if ori_size:
        n = ori_size
    else:
        n = np.size(y_true)
    return np.sum(np.abs(y_true - y_pred))/n

def mape(y_true, 
         y_pred, 
         ori_size=None
         )->float:
    """
        Returns the mean absolute percentage error between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        ori_size : int, default=None
            Optional. The number of values in each arrays.

        Returns
        ----------
        mape : float
            The MAPE between y_true and y_pred.
    """
    if ori_size:
        n = ori_size
    else:
        n = np.size(y_true)
    return np.sum(np.abs(y_true - y_pred)/y_true)/n

def positive_mae(y_true, 
                 y_pred, 
                 ori_size=None,
                 )->float:
    """
        Returns mean absolute error of predictions above the ground truth (over-prediction).
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        ori_size : int, default=None
            Optional. The number of values in each arrays.

        Returns
        ----------
        pmae : float
            The mean absolute error of predictions greather than the labels.
    """
    if ori_size:
        n = ori_size
    else:
        n = np.size(y_true)
        
    err = y_true - y_pred
    err_p = err[err<0]
    pmae = np.sum(err_p)/n
    return pmae

def relative_abs_err(y_true, 
                     y_pred,
                     ori_size=None,
                     )->float:
    """
        Returns the percentage relative error between true and predicted values.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.
        
        ori_size : int, default=None
            Optional. The number of values in each arrays.

        Returns
        ----------
        r_err : float
            The percentage relative error between y_true and y_pred.
    """
    if ori_size:
        n = ori_size
    else:
        n = np.size(y_true)
        
    err = 100*np.abs(y_true - y_pred)/y_true
    return np.sum(err)/n

def convergence(y_true, 
                y_pred
                )->float:
    """
        Returns the distance between (0,0) and the centroid of the curve of the absolute error curve, for a single unit.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        Returns
        ----------
        cv : float
            The convergence value with right to the absolute error function.
    """         
    abs_err = np.abs(y_true - y_pred)
    t = np.arange(0,len(y_true))
    total_area = np.sum((t[1:]-t[:-1]) * abs_err[:-1])
    xc = 0.5*np.sum((t[1:]**2 - t[:-1]**2) * abs_err[:-1]) / total_area
    yc = 0.5*np.sum((t[1:] - t[:-1]) * abs_err[:-1]**2) / total_area
    return np.sqrt(xc**2+yc**2)

def sample_std(y_true, 
               y_pred
               )->float:
    """
        Returns the sample standard deviation of the absolute error for a single unit.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        Returns
        ----------
        sstd : float
            The sample std.
    """    
    abs_err = np.abs(y_true - y_pred)
    return np.std(abs_err)

def relative_time_at_95(y_true, 
                        y_pred
                        )->float:
    """
        Returns the averaged % time after which 95% of the total error has been reached.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        Returns
        ----------
        rt95 : float
            The relative time at 95% of total error.
    """            
    cum_abs_err = np.cumsum(np.abs(y_true - y_pred))
    rt95 = np.argwhere(cum_abs_err>=0.95*cum_abs_err[-1])[0] / len(y_true)
    return rt95

def inference_time(model, 
                   data_type: str
                   )->float:
    """
        Computes and returns the inference time over the chosen dataset.
        The averaged time is returned in the case of training and validation data.

        Parameters
        ----------
        model : RULModel
            The RUL prediction model.

        data : str
            The data type to test. Can be either "test" or "test_full".

        Returns
        -------
        t : float
            The time in seconds to call model.predict().
    """

    if data_type=="test":
        x = model._test_dataset["data"]
    elif data_type=="test_full":
        x = model._test_full_dataset["data"]
    elif data_type=="train":
        x = model._train_datasets["fold_data"]
    elif data_type=="val":
        x = model._val_datasets["fold_data"]
    else:
        raise ValueError("Data type '{d}' not found.".format(d=data_type))

    if data_type in ["train", "val"]:
        start = time.time()
        for k in range(model._FOLDS):
            _ = model.predict(x[k], k)
        end = time.time()
        t = (end-start)/model._FOLDS
    else:
        start = time.time()
        _ = model.predict(x, -1, do_mean=True)
        end = time.time()
        t = end-start
    return t

def pearson_corr(y_true, 
                 y_pred,
                 ori_size=None
                 )->float:
    """
        Returns the pearson correlation coefficient between y_true and y_pred.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        ori_size : int, default=None
            Optional. The number of values in each arrays.

        Returns
        ----------
        p : float
            The pearson correlation coefficient.
    """            
    return pearsonr(y_true, y_pred).statistic