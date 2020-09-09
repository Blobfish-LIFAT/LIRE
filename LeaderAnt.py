"""
This package contains the fit function
"""
from math import floor
import numpy as np
from numpy.random import permutation

class LeaderAnt:

    def __init__(self, data):
        self.data = data
        self.n = self.data.shape[0]
        self.clusters = {}
        self.order = []

    def init_threshold(self, m, it):
        """
        Set the distance threshold to build clusters
        :param m:
        :param it:
        :return:
        """
        i = 0
        dist = 0
        while i < it:
            data1 = np.random.randint(0, self.n)
            data2 = np.random.randint(0, self.n)
            dist += m(self.data[data1], self.data[data2])
        return dist / it

    def dist_to_cluster(self, m, key, ind, comp):
        """
        Compute the distance between a data instance "ind" and a cluster "key" with metric m over "comp" comparisons
        :param m: metric
        :param key: the key of a cluster in the self.clusters dictionary
        :param ind: index of a specific data instance
        :param comp: number of comparisons. This is a max value since cluster may contain less than comp different points
        :return: the average distance to cluster "key"
        """
        c = self.clusters.get(key)  # retrieve list of instance index in cluster key
        it = min(comp, c.shape[0]) # estimating max number of comparisons

        # list of instances from cluster chosen at random to estimate distance
        subset = np.random.choice(c, it, replace=False)
        i = 0
        dist = 0
        for elem in subset:
            dist += m(self.data[elem], self.data[ind])
        return dist / iter

    def closest_cluster(self, ind, m, comp):
        """
        :param ind: index of instance to cluster
        :param m: metric
        :param comp: number of comparisons. This is a max value since cluster may contain less than comp different points
        :return: the key of the closest cluster and the distance to point ind
        """

        closest_cluster = float('inf')
        best_cluster = -1
        for k in self.clusters.keys():
            d = self.dist_to_cluster(m, k, ind, comp)
            if d < closest_cluster:
                closest_cluster = d
                best_cluster = k
        return best_cluster, closest_cluster

    def fit(self, m, iter_training=0.1, comp=5, epsilon=0.1):
        """
        :param m: metric used for distance computation
        :param iter_training: training set used to compute the distance threshold
        :param comp: max. number of comparisons of each point with each existing cluster
        :param epsilon: ratio of the dataset. A cluster above this size is removed from cluster list
        :return: set the value of clusters dictionary that indicates for each cluster a list of points assigned to
        this cluster
        """
        ## step 1. set the order of data instances randomly to simulate a stream
        self.order = permutation(self.n)

        ## step 2. init distance threshold
        t = self.init_threshold(m, floor(self.n * iter_training))

        ## step 3. build the cluster - pass one
        self.clusters = {0 : np.array([self.order[0]])} # 1st point => 1st cluster
        cluster_index = 1    # next cluster index to create
        for i in range(1, self.n):
            index, distance = self.closest_cluster(self.order[i], m, comp)
            if distance < t:
                # add to the closest cluster
                np.append(self.clusters.get(index), [self.order[i]])
            else:
                # create a new cluster
                self.clusters[cluster_index] = np.array([self.order[i]])
                cluster_index += 1  # ready for the next cluster!

        ## step 4. (optional) remove small clusters - pass two
        min_size = epsilon * self.n
        for k in self.clusters.keys():
            if self.clusters.get(k).shape[0] <= min_size:
                # todo: remove cluster
                # todo: reassign points to remaining clusters
        return 0