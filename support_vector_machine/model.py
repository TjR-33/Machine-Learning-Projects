import numpy as np
from tqdm import tqdm

class SVM_class:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        #implement this
        # initialize the parameters
        n_samples, n_features = X.shape
        # self.w = np.random.rand(n_features)
        # self.b = np.random.rand()
        self.w = np.zeros(n_features)
        self.b = 0
        pass
    
    
    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
            batch_fraction: float = 1.0
    ) -> None:
        self._initialize(X)
        #implement this
        # fit the SVM model using stochastic gradient descent
        # for i in tqdm(range(1, num_iters + 1)):
        #     # sample a random training example

        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        # progress_bar = tqdm(range(1, num_iters + 1))
        # for i in progress_bar:
        for i in tqdm(range(1, num_iters + 1)):
            term_w=np.zeros(n_features)
            term_b=0

            # batch = np.random.randint(0, n_samples, size=n_samples//2)
            if int(n_samples*batch_fraction)<=0:
                batch = np.random.choice(range(n_samples), size=1, replace=False)
            else:
                batch = np.random.choice(range(n_samples), size=int(n_samples*batch_fraction), replace=False)
            for idx in batch:
                x_i = X[idx]
                y_i = y[idx]
                if y_i == 0:
                    y_i = -1
                condition = 1 - y_i * (np.dot(self.w, x_i) + self.b)
                if condition > 0:
                    term_w += y_i*x_i
                    term_b += y_i
            self.w = self.w*(1-learning_rate) + learning_rate*C*term_w
            self.b = self.b + learning_rate*C*term_b  
        return None

        raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        return np.where(np.dot(X, self.w) + self.b > 0, 1, 0)
        raise NotImplementedError
    
    def accuracy_score(self, X, y) -> float:
        #implement this
        return np.sum(self.predict(X) == y) / len(y)
        raise NotImplementedError
    
    def precision_score(self, X, y) -> float:
        #implement this
        true_positive = np.sum((self.predict(X) == 1) & (y == 1))
        false_positive = np.sum((self.predict(X) == 1) & (y == 0))
        denominator = true_positive + false_positive
        if denominator == 0:
            return 0.0
        return true_positive / denominator
        raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        #implement this
        true_positive = np.sum((self.predict(X) == 1) & (y == 1))
        false_negative = np.sum((self.predict(X) == 0) & (y == 1))

        denominator = true_positive + false_negative
        if denominator == 0:
            return 0.0
        
        return true_positive / denominator
        raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        #implement this
        precision = self.precision_score(X, y)
        recall = self.recall_score(X, y)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
        raise NotImplementedError



    
