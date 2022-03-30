# K_Means
class KMeans:
    
    def __init__(self, k, method='random', max_iter=300):
        self.k = k 
        self.method = method
        self.max_iter = max_iter
        pass
    
    def init_centers(self, X):
        if self.method == 'random':
            return X[np.random.choice(X.shape[0], self.k, replace=False)]
        if self.method == 'k-means++':
            centroids = X[np.random.choice(X.shape[0], 1)]
            for k in range(self.k-1):
                dists = []
                for centroid in centroids:
                    dist = [np.linalg.norm(x - centroid) for x in X]
                    dists.append(dist)
                min_dist = [min(i) for i in zip(*dists)]
                probs = [i/sum(min_dist) for i in min_dist]
                centroids = np.vstack((centroids, X[np.random.choice(X.shape[0], 1, p=probs)]))
            return centroids
            
    def fit(self, X):
        self.centroids = self.init_centers(X)
        for _ in range(self.max_iter):
            clusters = self.expectation(X, self.centroids)
            new_centroids = self.maximization(X, clusters)
            if (new_centroids == self.centroids).all():
                break
            self.centroids = new_centroids
            
    def expectation(self, X, centroids):
        clusters = [[] for i in range(self.k)]
        for x in X:
            dist_dict = dict()
            for i, centroid in enumerate(centroids):
                dist_dict[i] = np.linalg.norm(x-centroid)
            sorted_dists = sorted(dist_dict.items(), key=lambda item: item[1])
            clusters[sorted_dists[0][0]].append(x)
        clusters = np.array([np.array(cluster) for cluster in clusters], dtype = object)
        return clusters

    def maximization(self, X, clusters):
        new_centroids = self.centroids.copy()
        for i, cluster in enumerate(clusters):
            new_centroids[i] = cluster.mean(axis = 0)
        return new_centroids
        
    def predict(self, X):
        predictions = []
        for x in X:
            dist_dict = dict()
            for i, centroid in enumerate(self.centroids):
                dist_dict[i] = np.linalg.norm(x-centroid)
            predictions.append(sorted(dist_dict.items(), key=lambda item: item[1])[0][0])
        return np.array(predictions)
    
    def predict_proba(self, X):
        probas = []
        for x in X:
            dists = [np.linalg.norm(x-centroid) for centroid in self.centroids]
            dists = np.array(dists, dtype=float)
            sum_dists = 0
            zero = []
            for i in range(len(dists)):
                if dists[i] != 0:
                    dists[i] = 1 / dists[i]
                    sum_dists += dists[i]
                else:
                    zero.append(i)
            dists /= sum_dists
            for i in zero:
                dists[i] = 1
            proba = dict()
            for i in range(len(self.centroids)):
                proba[i] = dists[i] / sum(dists)
            probas.append(proba)
        return np.array(probas)
