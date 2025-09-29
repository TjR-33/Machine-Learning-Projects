import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SkPCA
from model import PCA as MyPCA
import timeit
import pickle

# Load Data
X1 = np.load("data_matrix.npy")
shape1 = (256, 256, 4)
print(f"Data matrix with Opacity, loaded with shape {X1.shape}\n")

# Load PCA
skpca = SkPCA()
mypca = MyPCA()

print("Performing PCA on Data matrix\n")
print("Sklearn PCA")
start = timeit.default_timer()
skpca.fit(X1)
stop = timeit.default_timer()
print(f"Time: {stop - start}s")
print(f"Saving Sklearn PCA model...")

with open('skpca.pkl', 'wb') as f:
    pickle.dump(skpca,f)

print(f"Model saved\n")

print("MyPCA")
start = timeit.default_timer()
mypca.fit(X1)
stop = timeit.default_timer()

print(f"Time: {stop - start}s")
print(f"Saving MyPCA model...")

with open('mypca.pkl', 'wb') as f:
    pickle.dump(mypca,f)

print(f"Model saved\n")
