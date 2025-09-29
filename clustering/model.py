import numpy as np
# from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=10):
        self.k = k
        self.max_iter = max_iter

    def initialize(self, X):
        self.responsibility_matrix = np.zeros((X.shape[0], self.k))
        self.weights = np.ones(self.k) / self.k
        # min_vals = np.min(X)
        # max_vals = np.max(X)
        self.mu = X[np.random.choice(X.shape[0], size=(self.k,), replace=False)]
        #self.mu = np.random.uniform(min_vals, max_vals, size=(self.k, X.shape[1]))
        self.sigma = np.array([np.cov(X.T) + np.eye(X.shape[1]) * 1e-6 for _ in range(self.k)])
       

    def e_step(self, X):
        # denominator = 0
        for j in range(self.k):
            sigma_j = self.sigma[j] + np.eye(self.sigma[j].shape[0]) * 1e-6
            # self.responsibility_matrix[:,j] = self.weights[j] * multivariate_normal.pdf(X, self.mu[j], sigma_j)
            self.responsibility_matrix[:,j] = self.weights[j] * self.multivariate_normal(X, self.mu[j], sigma_j)
        self.responsibility_matrix /= np.sum(self.responsibility_matrix, axis=1)[:, np.newaxis]

        
    def m_step(self, X):
        total_weight = np.sum(self.responsibility_matrix, axis=0)
        self.weights = total_weight / X.shape[0]
        self.mu = np.dot(self.responsibility_matrix.T, X) / total_weight[:, np.newaxis]
        self.sigma = []
        for k in range(self.k):
            diff = (X - self.mu[k]).T
            sigma_k = np.dot(self.responsibility_matrix[:, k] * diff, diff.T) / total_weight[k]
            self.sigma.append(sigma_k)
        self.sigma = np.array(self.sigma)
        # for k in range(self.k):
        #     N_k = np.sum(self.responsibility_matrix[:, k])
        #     mu_numerator = np.zeros(self.mu[0].shape)
        #     sigma_numerator = np.zeros(self.sigma[0].shape)
        #     for m in range(X.shape[0]):
        #         mu_numerator = mu_numerator + self.responsibility_matrix[m, k] * X[m]
        #         sigma_numerator = sigma_numerator + self.responsibility_matrix[m, k] * (X[m] - self.mu[k]) @ (X[m] - self.mu[k]).T


        #     self.sigma[k] =  np.diag(np.diag(sigma_numerator / N_k))
        #     # self.sigma[k] = sigma_numerator / N_k


    def fit(self, X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def multivariate_normal(self,x,mu,cov):
            d = mu.shape[0]

            cov_det = np.linalg.det(cov)
            cov_inv = np.linalg.inv(cov)

            norm_const = 1.0 / (np.power((2 * np.pi), d / 2) * np.sqrt(cov_det))

            x_mu = np.array(x - mu)
            exponent = -0.5 * np.einsum('ij,ij->i', x_mu, np.dot(cov_inv, x_mu.T).T)

            return norm_const * np.exp(exponent)

    def predict_proba(self, X):
        # find posterior probability of each x_i given the data
        # p(z_i| x_i, mu, sigma)
        # matrix = np.zeros((X.shape[0], self.k))
        # for m,x_m in enumerate(X):
        #     denominator = 0
        #     for j in range(self.k):
        #         t = self.weights[j] * self.multivariate_normal(x_m, self.mu[j], self.sigma[j])
        #         denominator += t
        #         matrix[m, j] = t
        #     matrix[m, :] /= denominator
        # pass
        # print(matrix)
        gamma = np.zeros((X.shape[0], self.k))
        for j in range(self.k):
            # gamma[:,j] = multivariate_normal.pdf(X, self.mu[j], self.sigma[j])
            gamma[:,j] = self.multivariate_normal(X, self.mu[j], self.sigma[j])
                # t= t + 1e-10
                # denominator += t 
        gamma = gamma * self.weights
        gamma /= np.sum(gamma, axis=1)[:, np.newaxis]
        return gamma

    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)

class KMeans:
    def __init__(self, k, max_iter=10):
        self.k = k
        self.max_iter = max_iter

    def initialize(self, X):
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        # print(min_vals, max_vals)
        self.centroids = X[np.random.choice(X.shape[0], size=(self.k,), replace=False)]
        # self.centroids = np.random.uniform(min_vals, max_vals, size=(self.k, X.shape[1]))        
        self.cluster = np.zeros(X.shape[0]) 

    def fit(self, X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            # assigning each data point to the closest centroid
            for n in range(X.shape[0]):
                distances = np.array([np.linalg.norm(X[n] - c) for c in self.centroids])
                self.cluster[n] = np.argmin(distances,)
            # calculating new centroid for each cluster
            for m in range(self.k):
                if np.sum(self.cluster == m) > 0:
                    self.centroids[m] = np.mean(X[self.cluster == m], axis=0)

    def predict(self, X):
        # predictions = np.array([np.argmin(np.array([np.linalg.norm(x - c) for c in self.centroids])) for x in X])
        predictions = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            distances = np.array([np.linalg.norm(X[n] - c) for c in self.centroids])
            predictions[n] = np.argmin(distances)
        return predictions
	
