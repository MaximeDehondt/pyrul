# tsbpdtw_core.py
# Architecture of the TSBP model.
# Code mostly taken from https://github.com/huanglinxixi/DTW_RUL_Trajectory/blob/master/main.py
import numpy as np
import heapq
import tslearn
from tslearn import metrics

class TSBP:
    def __init__(self, 
                 dtw_tol: int, 
                 nb_sim: int, 
                 k: int, 
                 d1: int, 
                 d2: int, 
                 r_early: float, 
                 time_factor : float
                 ):
        self._dtw_tol = dtw_tol
        self._nb_sim = nb_sim
        self._k = k
        self._D1 = d1
        self._D2 = d2
        self._R_EARLY = r_early
        self._time_factor = time_factor # model assumes regular time steps, from 1 to len(trajectory) included. This attribute acts as a factor.

    def fit(self, 
            x: np.array, 
            y: np.array
            )->None:
        """
            Saves the given train datasets as a set of trajectories history.
            
            Parameters
            ----------
            x : list of 1d np.array
                The train dataset of trajectories.

            y : list of 1d np.array
                The respective RUL labels.
        """
        self._train_data = x
        self._train_labels = y

    def predict(self, 
                test_data: list
                )->np.array:
        """
            Computes and returns the predicted RUL values of each given test trajectory, at their last known time step.
            
            Parameters
            ----------
            test_data : list of 1d np.array
                The test dataset of trajectories.

            Returns
            -------
            RUL_pred : list of int
                The predicted RUL values.
        """
        # For each test trajectories, computes DTW with all train trajectories
        n = len(test_data)
        min_distance = [] # list of list of dtw values
        for i in range(n):
            dist_min_all = self.similarity(test_data[i], k3=self._dtw_tol)
            min_distance.append(dist_min_all)
            
        RUL = []
        for i in range(n):
            RUL_temp = self.simple_RUL_trajectory(test_traj=test_data[i], 
                                                  top_numbers=self._nb_sim, 
                                                  id_min_distance=min_distance[i], 
                                                  k=self._k)
            RUL.append(RUL_temp)
        RUL_pred = [round(i,1) for i in RUL]
        return np.array(RUL_pred)


    def similarity(self, 
                   test_traj: np.array, 
                   k3: int
                   )->list:
        """
            Computes the DTW between test_traj and all other train trajectories (that can be slightly expanded or shrunk to minimize the distance).

            Parameters
            ----------
            test_traj : 1d np.array
                The test trajectory.

            k3 : int
                The maximum step to expand or shrunk the train trajectories.

            Returns
            -------
            min_distance : list of float
                The minimal DTW distances for each train trajectories w.r.t. the test trajectory.
        """
        min_distance = []
        n = len(self._train_data)
        n_test = len(test_traj)
        intervals = np.arange(-k3, k3 + 1)
        
        for i in range(n):
            train_traj = self._train_data[i]
            n_train = len(train_traj)
            distance_temp = []

            # Computing simple DTW if test segment is too long even if n_train is expanded
            if (n_test >= n_train + k3):
                _, d = tslearn.metrics.dtw_path(train_traj, test_traj)
                distance_temp.append(d)
            else:
                # Otherwise we allow train segments to be expanded or shrunk up to k3 values (when possible),
                # so as to get the most relevant train segment portion
                for interval in intervals:
                    adjusted_length = n_test + interval
                    
                    # Only compute DTW if within bounds
                    if (adjusted_length > 0) and (adjusted_length <= n_train):
                        train_traj_temp = train_traj[:adjusted_length]
                        _, d = tslearn.metrics.dtw_path(train_traj_temp, test_traj)
                        distance_temp.append(d)
                        
            min_distance_temp = min(distance_temp)
            min_distance.append(min_distance_temp)
        return min_distance
    
    def simple_RUL_trajectory(self, 
                              test_traj: np.array, 
                              top_numbers: int, 
                              id_min_distance, 
                              k: int
                              )->np.array:
        """
            Predicts the last RUL value of the test trajectory for each reference train trajectories
            and returns an interpolated value of the maximum and minimum RULs.
            Returned value is between 0 and self._R_EARLY (if not None).

            Parameters
            ----------
            test_traj : 1d np.array
                The test trajectory.

            top_numbers : int
                The number of similar train trajectories to keep at most.

            id_min_distance : list
                The list of the DTW values for each train trajectory.

            k : float
                The interpolation factor, to put more weight on RUL_train_min (k=1) or RUL_train_max (k=0).

            Returns
            -------
            RUL : float
                The estimated RUL value.
        """
        dist_min_id, dist_min = self.outlier_remove(test_traj=test_traj, 
                                                    top_numbers=top_numbers,
                                                    id_min_distance=id_min_distance)
        RUL_train = []
        for i in range(len(dist_min_id)):
            rul_i = (len(self._train_data[dist_min_id[i]]) - len(test_traj)) * self._time_factor
            RUL_train.append(rul_i)

        RUL = k * min(RUL_train) + (1-k) * max(RUL_train)
        
        if self._R_EARLY is not None:
            RUL = np.clip(RUL, 0, self._R_EARLY)
        else:
            RUL = np.maximum(0, RUL)
            
        return RUL

    def outlier_remove(self, 
                       test_traj: np.array, 
                       top_numbers: int, 
                       id_min_distance: list
                       )->list:
        """
            Keeps the top_numbers-smallest dtw values in id_min_distance, filters out train trajectories
            with inappropriate total life values and removes statistical outliers.

            Parameters
            ----------
            test_traj : 1d np.array
                The test trajectory.

            top_numbers : int
                The number of similar train trajectories to keep at most.

            id_min_distance : list
                The list of the DTW values for each train trajectory.

            Returns
            -------
            dist_min_id : list of int
                The indices of the most similar (w.r.t. the test trajectory) train trajectories to keep.
                Contains at least one value, and at most top_numbers values.

            dist_min : list of float
                The respective DTW values.
        """
        # Computing the training trajectories indices associated to the top_numbers-smallest dtw values
        min_number = heapq.nsmallest(top_numbers, id_min_distance) # min_number contains the n-smallest DTW values, not indices
        dist_min_id = []
        for i in range(top_numbers): # search of the indices
            dist_min_id.append(id_min_distance.index(min_number[i]))

        # Delete those with a total life discrepancy greater than D1 or less than D2
        id_to_delete = []
        n_test = len(test_traj)
        for i in range(len(dist_min_id)):
            n_train = len(self._train_data[dist_min_id[i]])
            diff = n_train - n_test

            if diff >= self._D1: # if train data is too big compared to test data
                id_to_delete.append(dist_min_id[i])
            elif n_train < self._D2: # if train data is too small
                id_to_delete.append(dist_min_id[i])

        save_idx = (len(dist_min_id)==len(id_to_delete)) # to avoid empty dist_min_id, we keep at least 1 idx
        for i in range(len(id_to_delete)-save_idx):
            dist_min_id.remove(id_to_delete[i])

        # Median removal if possible
        if len(dist_min_id)>=2:
            RUL_temp = []
            for i in range(len(dist_min_id)):
                n_train = len(self._train_data[dist_min_id[i]])
                diff = n_train - n_test
                RUL_temp.append(diff)
        
            RUL_temp.sort()
            RUL_25 = np.median(RUL_temp[:len(RUL_temp) // 2])
            RUL_50 = np.median(RUL_temp)
            RUL_75 = np.median(RUL_temp[len(RUL_temp) // 2:])
        
            id_to_delete = []
            for i in range(len(dist_min_id)):
                if (RUL_50 - 3 * (RUL_50 - RUL_25)) > RUL_temp[i]:
                    id_to_delete.append(dist_min_id[i])
                if (RUL_50 + 2 * (RUL_75 - RUL_50)) < RUL_temp[i]:
                    id_to_delete.append(dist_min_id[i])

            save_idx = (len(dist_min_id)==len(id_to_delete)) # to avoid empty dist_min_id, we keep at least 1 idx
            for i in range(len(id_to_delete)-save_idx):
                dist_min_id.remove(id_to_delete[i])

        dist_min = [id_min_distance[d] for d in dist_min_id]

        return dist_min_id, dist_min