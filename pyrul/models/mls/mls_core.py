# mls_core.py
# Architecture of the MLS model.
import numpy as np
from dtw import *
from tqdm import tqdm

class MLS:
    """
            Main Multiple Local Similarties model.
    """
    def __init__(self,
                 window: int,
                 sliding_step: int,
                 k: int, 
                 delta_w: int, 
                 alpha: float, 
                 beta: float, 
                 lambda1: float, 
                 lambda2: float, 
                 h: float,
                 r_early: float,
                 time_factor: int):
        
        self._window = window
        self._sliding_step = sliding_step
        self._k = k
        self._delta_w = delta_w
        self._alpha = alpha
        self._beta = beta
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self._h = h
        self._r_early = r_early
        self._time_factor = time_factor # model assumes regular time steps, from 1 to len(trajectory) included. This attribute acts as a factor.

        # Data history
        self._x_hist = None
        self._y_hist = None
        self._x_hist_seg = None
        self._y_hist_seg = None
        self._data_hist = None

    def get_expanded_train_segment(self, 
                                   idx: int, 
                                   p: int, 
                                   seg: np.array
                                   )->np.array:
        """
            Returns the idx-th train segment expanded by p values if possible.
        
            Parameters
            ----------
            idx : int
                The idx of the train segment in self._x_hist_seg.
                
            p : int
                The number of values to add to the segment.
                
            seg : np.array of shape (nb_values)
                The segment to expand (can be a shrunken or expanded version)

            Returns
            -------
            new_segment : np.array of shape (nb_values)
                The new segment.
        """
        # We expand the idx-th segment using the original trajectory it belongs too
        traj_idx, inner_idx = self._data_hist[idx]
        return self._x_hist[traj_idx][inner_idx*self._sliding_step:inner_idx*self._sliding_step+len(seg)+p]
    
    def get_expanded_test_segment(self, 
                                   idx: int, 
                                   p: int, 
                                   test_system: np.array
                                   )->np.array:
        """
            Returns the idx-th test segment expanded by p values if possible.
        
            Parameters
            ----------
            idx : int
                The idx of the test segment in the list of test segments.
                
            p : int
                The number of values to add to the segment.
                
            test_system : np.array of shape (nb_values)
                The initial test trajectory.

            Returns
            -------
            new_segment : np.array
                The new segment.
        """
        return test_system[idx*self._sliding_step:idx*self._sliding_step+self._window+p]
    
    def get_last_ref_cycle(self, 
                           idx: int, 
                           ref_seg: np.array
                           )->float:
        """
            Returns the last cycle of the reference segment using its original index. 
            The segment can be a shrunken or extended version of itself.

            Parameters
            ----------
            idx : int
                The index of the segment in self._data_hist
                    
            ref_seg : np.array
                The reference segment.

            Returns
            -------
            t : float
                The last time step value of the reference segment.
        """
        _, inner_idx = self._data_hist[idx]
        return (inner_idx*self._sliding_step+len(ref_seg)) * self._time_factor
          
    def fit(self, 
            x_hist: list, 
            y_hist: list
            )->None:
        """
            Defines the processed historical system T of the model.
            
            Parameters
            ----------
            x_hist : list of 1d np.array
                    Contains every historical systems of variable length.
                    
            y_hist : list of 1d np.array
                    Contains every historical systems's RULs.
        """
        # We keep the original trajectories and we keep the DI segmentations of them
        # because we will need to shrink or expand some segments
        self._x_hist = x_hist
        self._y_hist = y_hist
        self._x_hist_seg = []
        self._y_hist_seg = []
        
        # DI Segmention (we keep track of in which trajectory and where in it the segment is)
        self._data_hist = []
        for i in range(len(x_hist)):
            x, y = self._x_hist[i], self._y_hist[i]
            
            seg_range = np.arange((len(x)-self._window)//self._sliding_step + 1, dtype='int32') # contains the idx of all segments inside the i-th trajectory
            self._data_hist.append((np.vstack([[i]*len(seg_range), seg_range]).T))
            self._x_hist_seg += [x[self._sliding_step*k:self._sliding_step*k+self._window] for k in seg_range] # 1d segment
            self._y_hist_seg += [y[self._sliding_step*k+self._window-1] for k in seg_range] # last associated rul value
        self._data_hist = np.vstack(self._data_hist).astype(dtype='int32') # shape : (total_nb_windows, 2)
        # 1 element of self._data_hist contains  (trajectory idx, position of segment in trajectory)
        
    def __call__(self, 
                 test_system, 
                 t_pred: int
                 )->float:
        """
            Parameters
            ----------
            segments_test : np.array of shape (nb_windows, window_size)
                    Contains every test window
            t_pred : int
                    The desired prediction time.
                    
            Returns the estimated expected RUL for the test system at desired prediction time using processed historical systems.
        """
        # DI Segmentation of test system
        seg_range = np.arange((len(test_system)-self._window)//self._sliding_step + 1, dtype='int32')
        segments_test = [test_system[self._sliding_step*k:self._sliding_step*k+self._window] for k in seg_range]
        test_cycles = np.arange(1,len(test_system)+1, dtype='int32') * self._time_factor 
        last_test_cycles = [test_cycles[self._sliding_step*k+self._window-1] for k in seg_range]
        
        n = len(segments_test)
        ruls_ref_set = np.empty(self._k*n) # reference ruls
        segs_ori_idx = np.empty(self._k*n, dtype='int32') # keeping track of original segments's index
        segments_ref_set = [] # list of reference segments that will contain self._k * n segments
        
        # For each test segment
        for i, s_test in enumerate(segments_test):
            # Similar reference segments selection
            segments_ref, ruls_ref, seg_ori_idx = self.sim_ref_seg_selection(s_test) # segments_ref arrays shape : (self._k, +/- window_size)

            # Degradation rate adjustment
            ruls = self.de_rate_adjust(np.array(ruls_ref), segments_ref, s_test)
            
            # Time gap adjustment
            ruls = self.time_gap_adjust(ruls, t_pred, last_test_cycles[i])
            
            # Saving results
            segments_ref_set += segments_ref
            ruls_ref_set[i*self._k:(i+1)*self._k] = ruls
            segs_ori_idx[i*self._k:(i+1)*self._k] = seg_ori_idx
        
        # Computing weights for each reference segment
        weights = np.empty(self._k*n)
        for i in range(self._k*n):

            weights[i] = self.self_adaptive_weight_assignement(s_test=segments_test[i//self._k], 
                                                               idx_s_test=i//self._k, 
                                                               t_test_last=last_test_cycles[i//self._k], 
                                                               s_ref=segments_ref_set[i],
                                                               idx_s_ref=segs_ori_idx[i],
                                                               t_ref_last=self.get_last_ref_cycle(segs_ori_idx[i], segments_ref_set[i]),
                                                               test_system=test_system, 
                                                               t_pred=t_pred)

        if np.sum(weights)>0.:
            weights /= np.sum(weights) # normalize each weight
            
        return np.sum(ruls_ref_set * weights) #lambda x: kernel_density_estimation(x, weights, ruls_ref) to return a KDE of the rul of the test systems at the desired prediction time
            
    def sim_ref_seg_selection(self, 
                              s: np.array
                              )->tuple:
        """
            Returns 3 arrays containing the K most similar train windows to s (they may be shrunk or extended), 
            their respective RUL, and their original index in self._x._hist_seg.
            
            Parameters
            ----------
            s : np.array of shape (window_size)
                A window.

            Returns
            -------
            x_ref : list of np.arrays of shape (K, window_size)
                The K most similar (variable length) windows.

            y_ref : list
                The related RUL values.

            smallest_idx : np.array
                The original indices of the related windows in self._x_hist_seg
            
        """
        # Coarse search phase : finding the K most similar train segments
        distances = np.array([dtw(s, x).distance for x in self._x_hist_seg])
        smallest_idx = np.argpartition(distances, self._k)[:self._k]
        
        # Fine search phase : replacing previous segments by modified versions of them that are most similar to the test segment
        x_ref, y_ref = [], []
        for i in smallest_idx:
            # We extend or shrink each reference segment whenever possible, using self._x_hist
            sub_x_hist, sub_y_hist = [], []
            for p in range(-self._delta_w, self._delta_w+1):
                if p<0:
                    # Shrinking the view of the segment
                    sub_x_hist.append(self._x_hist_seg[i][:p])
                    # Adjusting the RUL of the shrunk segment
                    new_rul = self._y_hist_seg[i] - p * self._time_factor
                    if self._r_early is not None:
                        new_rul = np.minimum(self._r_early, new_rul)
                    sub_y_hist.append(new_rul)
                elif p>0:
                    # Extending the segment
                    sub_x_hist.append(self.get_expanded_train_segment(i, p, self._x_hist_seg[i]))
                    # Adjusting the RUL of the extended segment
                    new_rul = self._y_hist_seg[i] - p * self._time_factor
                    if self._r_early is not None:
                        new_rul = np.minimum(self._r_early, new_rul)
                    sub_y_hist.append(new_rul)
                else:
                    sub_x_hist.append(self._x_hist_seg[i])
                    sub_y_hist.append(self._y_hist_seg[i])
            
            # Finding the smallest distance for the current reference segment
            distances = np.array([dtw(s, x).distance for x in sub_x_hist])
            smallest_idx_bis = np.argmin(distances)
            x_ref.append(sub_x_hist[smallest_idx_bis])
            y_ref.append(sub_y_hist[smallest_idx_bis])
            
        return x_ref, y_ref, smallest_idx
    
    def s_fun(self, 
              x: float
              )->float:
        """
            The scaling s function.

            Parameters
            ----------
            x : float
                The input value.

            Returns
            -------
            res : float
                The output value.
        """
        return np.tanh(x)
    
    def adjusting_fun(self, 
                      x: float
                      )->float:
        """
            The adjusting function.

            Parameters
            ----------
            x : float
                The input value.

            Returns
            -------
            res : float
                The output value.
        """
        factors = (x>=0) * self._alpha + (x<0) * self._beta
        return 1 - self.s_fun(factors * x)

    def de_rate_adjust(self, 
                       y_ref: np.array, 
                       x_ref: list, 
                       x: np.array
                       )->float:
        """
            Ajdusts y_ref based on the difference of degradation between x_ref and x segments.
            The adjusting function.

            Parameters
            ----------
            y_ref : np.array
                The input RUL values to adjust.

            x_ref : list of np.array
                The related reference segments.

            x : np.array
                The test segment.

            Returns
            -------
            res : np.array
                The adjusted reference RUL values.
        """
        de_rate_s_ref = np.array([(s[-1] - s[0]) / (len(s)*self._time_factor) for s in x_ref])
        de_rate_test = (x[-1] - x[0]) / (len(x)*self._time_factor)
        return self.adjusting_fun(de_rate_test - de_rate_s_ref) * y_ref
    
    def time_gap_adjust(self, 
                        ref_rul: np.array, 
                        t_pred: int, 
                        t_last: int
                        )->np.array:
        """
            Adjusts the estimated rul_s_ref to the prediction t_pred
            
            Parameters
            ----------
            ref_rul : 1d np.array
                Contains all estimated RULs.
                
            t_pred : int
                The desired prediction time.
                
            t_last : int
                The last cycle of the current test segment.

            Returns
            -------
            res : np.array
                The adjusted referenced RULs.
        """
        tg = t_pred - t_last # time gap between last cycle of s_test and cycle of t_pred
        return ref_rul - tg
    
    def self_adaptive_weight_assignement(self, 
                                         s_test, 
                                         idx_s_test, 
                                         t_test_last, 
                                         s_ref, 
                                         idx_s_ref, 
                                         t_ref_last, 
                                         test_system, 
                                         t_pred):
        """
            Computes a raw weight importance of the given reference segment, w.r.t. the test segment and the desired prediction time.
            The weight is w=w1*w2 with :
                - w1: how much the reference segment is similar to the test one.
                - w2: how much the reference segment's future evolution is similar to that of the test segment.

            Parameters
            ----------
            s_test : 1d np.array
                The test segment.
                
            s_ref : 1d np.array
                The reference segment.
                
            test_system : 1d np.array
                The test system.
                
            t_pred : int
                The prediction time.
                
            t_test_last : int
                The last cycle of the test segment.
                
            t_ref_last : int
                The last cycle of the reference segment.

            Returns
            -------
            w : float
                The non-normalized weight of the reference segment.
        """
        # Several scenarios depending on whether there exists a common future period
        tg = t_pred - t_test_last
        bias = 0
        if tg>0:
            s_test_future = self.get_expanded_test_segment(idx_s_test, tg//self._time_factor, test_system) # we expand by tg values
            s_ref_future = self.get_expanded_train_segment(idx_s_ref, tg//self._time_factor, s_ref) # we TRY to expand by tg values
            actual_nb_new_vals = len(s_ref_future) - len(s_ref)
            
            if actual_nb_new_vals!=0:
                common_period = min(tg//self._time_factor, actual_nb_new_vals)
                s_ref_future = s_ref_future[-common_period:]
                s_test_future = s_test_future[-common_period:]
                bias = np.mean(np.abs(s_ref_future - s_test_future))

        w2 = np.exp(-self._lambda2 * bias)
        w1 = np.exp(-self._lambda1 * dtw(s_test, s_ref).distance)
        return w1 * w2
    
    def kernel(self, 
               x: float
               )->float:
        """
            Returns the kernel value at x.

            Parameters
            ----------
            x : float
                The input value.

            Returns
            -------
            res : float
                The kernel output value.
        """
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)
    
    def kernel_density_estimation(self, 
                                  rul: float, 
                                  weights: np.array, 
                                  ref_ruls: np.array
                                  )->float:
        """
            Returns the kernel density estimation at x=rul.
        
            Parameters
            ----------
            rul : int
                The RUL value to estimate the KDE at.
                
            weights : 1d np.array
                The list of normalized weights associated to each reference RUL.
                
            ref_ruls : 1d np.array
                The reference RULs.
                
            Returns
            -------
            res : float
                The KDE value at x=rul.
        """
        return np.sum(weights * self.kernel((rul - ref_ruls)/self._h))/self._h
    
    def predict(self, test_systems):
        """
            Returns the predicted RUL values for all test units, at their last recorded time step.

            Parameters
            ----------
            test_systems : list of np.array
                The test trajectories.

            Returns
            -------
            y_pred : np.array
                The predicted RUL values.
        """
        nb_traj = len(test_systems)
        y_pred = np.empty(nb_traj)
        for i in tqdm(range(nb_traj), desc="inference"):
            y_pred[i] = self(test_systems[i], len(test_systems[i])*self._time_factor)
        return y_pred