# viz.py
# Visualization functions.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyrul.models.base import RULModel
from pyrul.utils.various import tensor_to_numpy
from matplotlib.ticker import MaxNLocator

def plot_full_rul_pred(y, y_pred, cycles, save_path):
        
    plt.plot(cycles, y_pred, label="pred")
    plt.plot(cycles, y, label="truth")
    plt.legend()
    plt.grid()
    plt.xlabel("cycles")
    plt.ylabel("RUL")

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()


def plot_point_pred_err(y_true, y_pred, title=None, save_path=None):
    """
        Returns a plot of the prediction errors.
        
        Parameters
        ----------
        y_true : np.array of shape (n_values,)
            The ground truth labels.
    
        y_pred : np.array of shape (n_values,)
            The predicted labels.

        title : str (optional), default=None
            The name of the plot

        save_path : save_path (optional), default=None
            The path to save the plot.
            
        Returns
        ----------
        scoring : float
            The scoring function value between y_true and y_pred.

    """
    max_val = np.max(list(y_pred) + list(y_true))
    
    plt.scatter(y_true, y_pred, zorder=2)
    plt.grid()
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.ylim(-1,max_val+1)
    plt.plot([0,max_val+1],[0,max_val+1],'--',c='red')
    if title:
        plt.title(title)

    if save_path:
        plt.savefig(save_path)
        plt.clf()
    else:
        plt.show()

def plot_loss(histories, save_path: str, log_scale: bool=False):
    """
        Plots the evolution of the train MSE and val MSE for each model.

        save_path : str
            The path to save the plots.

        log_scale : bool, default=False
            Whether to use the log scale for the y-values.
    """
    nb_folds = len(histories)
    nb_epochs = len(histories[0].history['mse'])
    
    # Evolution of loss
    epoch_range = np.arange(1,nb_epochs+1)
    plt.figure(figsize=(7,4))
    for k in range(nb_folds):
    
        plt.plot(epoch_range, histories[k].history['mse'], label="train")
        plt.plot(epoch_range, histories[k].history['val_mse'], label="val")
        plt.grid()
        plt.xlabel("epochs")
        plt.ylabel("MSE")
        if log_scale:
            plt.yscale('log')
        plt.title("Model "+str(k+1))
        plt.legend()

        if save_path:
            path = save_path+"loss_evolution_model"+str(k)+".png"
            plt.savefig(path)
            plt.clf()
        else:
            plt.show()

def plot_pred(model: RULModel, 
              data_type: str,
              plot_indices: list, 
              fold_indices: list,
              model_indices: list,  
              save_path: str=None, 
              apply_threshold: bool=False,
              load_pred_paths: list=None,
              xlims: list=None,
              ylims: list=None,
              yticks: list=None):
    """
        For each given indices, plots the prediction of the desired model(s) on the specified dataset.

        model : RULModel
            The prediction model.

        data_type : str
            The dataset to choose. Is either "train", "val", "test_full" or "test_full_ind" (for individual fold prediction on full test data).

        plot_indices : list of ints
            The list of unit indices to choose from the dataset.

        fold_indices : list of ints
            The list of fold indices to choose from the dataset. Should be -1 if data_type=="test_full".

        model_indices : list of ints
            The list of model indices to choose for the predictions. 
            -1 calls all model to get an average prediction.

        save_path : str, default=None
            Optional. The path to save the plots.

        apply_threshold : bool, default=False
            Whether to apply the threshold on the true labels, if not already done.

        load_pred_paths : list of str, default=None
            Optional. If not None, the path to load the predictions (for each unit) will be used instead of calling model.predict().

        xlims : list of tuple of int, default=None
            Optional. The x limits of the plots for each unit.

        ylims : list of tuple of int, default=None
            Optional. The y limits of the plots for each unit.

        yticks : list of list of floats, default=None
            Optional. If not None, adjusts the plots to each desired yticks. Specified yticks must be in the range of respective ylims.
    """
    if load_pred_paths is None:
        load_pred_paths = [None]*len(plot_indices)
    if xlims is None:
        xlims = [None]*len(plot_indices)
    if ylims is None:
        ylims = [None]*len(plot_indices)
    if yticks is None:
        yticks = [None]*len(plot_indices)
    
    # Step 1. Loading data from model
    if data_type=="train":
        ds = model._train_datasets
    elif data_type=="val":
        ds = model._val_datasets
    elif data_type in ["test_full","test_full_ind"]:
        ds = model._test_full_dataset
    else:
        raise ValueError("Data type not found. Allowed values are 'train', 'val', and 'test_full'.")
    
    for k1, k2, idx, l_path, xlim, ylim, ytick in zip(model_indices, fold_indices, plot_indices, load_pred_paths, xlims, ylims, yticks):
    # Note : idx no longer corresponds to the same initial sample before k-fold

        # Step 2. Getting the data samples for the desired unit
        if data_type in ["test_full","test_full_ind"]:
            slice_inf, slice_sup = model.get_slices(idx, ds["lengths"], data_type="test_full")
            x = ds["data"]
            y = ds["labels"]
        else:
            slice_inf, slice_sup = model.get_slices(idx, ds["fold_lengths"][k2], data_type=data_type)
            x = ds["fold_data"][k2]
            y = ds["fold_labels"][k2]
            
        x = x[slice_inf:slice_sup]
        y = y[slice_inf:slice_sup]

        # Step 3. Loading or Computing predictions
        if l_path is not None:
            y_pred = np.loadtxt(fname=l_path+"_pred.txt", delimiter=",")

            if model._is_ibl and data_type in ["test_full","test_full_ind"]:
                a, b = model.get_slices(idx, ds["lengths"], data_type="test_full", is_pred_rul_compressed=True)
                y = y[0]
                y_pred = y_pred[a:b]
                y = np.array(y[model._MIN_TS_SIZE-1:])
            else:
                y_pred = y_pred[slice_inf:slice_sup]
                y, y_pred = tensor_to_numpy(y), tensor_to_numpy(y_pred)

            if model._is_deep_lstm:
                y = y[:,:,0]
                mask = y!=-1.
                y, y_pred = y[mask], y_pred[mask]

            y, y_pred = y.ravel(), y_pred.ravel()
            
        else:
            if model._is_ibl:
                # For a model that predicts the last RUL value at a desired time,
                # we call it for all possible timesteps.
                x, y = x[0], y[0]
                y_pred = np.empty(len(y)-model._MIN_TS_SIZE+1)
                for i in range(len(x)-model._MIN_TS_SIZE+1):
                    sub_x = x[:model._MIN_TS_SIZE+i]
                    y_pred[i] = model.predict([sub_x], k1)[0]
                y = y[model._MIN_TS_SIZE-1:]
                  
            else:                
                y_pred = model.predict(x, k1)
                y, y_pred = tensor_to_numpy(y), tensor_to_numpy(y_pred)
                y, y_pred = y.ravel(), y_pred.ravel()
                
            if model._is_deep_lstm:
                # For a model with masked data, we keep non-zero elements
                mask = y!=-1.
                y = y[mask]
                y_pred = y_pred[mask]

        if apply_threshold:
            y = np.minimum(y, model._R_EARLY)

        # Step 4. Getting the cycles of the unit
        if data_type in ["test_full","test_full_ind"]:
            cycles = model.get_cycles(length=ds["lengths"][idx], is_test=True)
        else:
            cycles = model.get_cycles(length=ds["fold_lengths"][k1][idx], is_test=False)

        # Step 5. Plotting
        plt.plot(cycles, y_pred, label=r'$\mathbf{\hat{y}}$', linewidth=3)
        plt.plot(cycles, y, label=r'$\mathbf{y}$', linewidth=3)
        plt.legend(fontsize=24, title_fontsize=18)
        plt.xlabel("cycles", fontsize=22, fontweight='bold')
        plt.ylabel("RUL", fontsize=22, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(ticks=ytick, fontsize=18)
        plt.grid(True)
            
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if save_path:
            path = save_path+"model_"+str(k1)+"_fold_"+str(k2)+"_pred_"+data_type+"_ts_"+str(idx)+".png"
            plt.savefig(path)
        else:
            plt.show()
        plt.clf()