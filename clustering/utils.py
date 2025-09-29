import numpy as np

def entropy(label_data):
   unique_labels, label_counts = np.unique(label_data, return_counts=True)
   probabilities = label_counts / len(label_data)
   return -np.sum(probabilities * np.log2(probabilities))

def mutual_information(true_labels, predicted_labels):
   clusters, cluster_counts = np.unique(predicted_labels, return_counts=True)

   mutual_info = 0
   for idx, cluster in enumerate(clusters):
      cluster_indices = np.flatnonzero(predicted_labels == cluster)
      mutual_info += entropy(true_labels[cluster_indices]) * cluster_counts[idx] / predicted_labels.size
   mutual_info -= entropy(true_labels)
   return -mutual_info

def calc_NMI(true_labels, predicted_labels):
   true_labels = np.array(true_labels).flatten()
   predicted_labels = np.array(predicted_labels).flatten()

   mutual_info = mutual_information(true_labels, predicted_labels)
   entropy_true = entropy(true_labels)
   entropy_predicted = entropy(predicted_labels)
   normalized_mutual_info = mutual_info / ((entropy_true + entropy_predicted) / 2.0  + 1e-10)

   return normalized_mutual_info


def read_data():   
   data_train = np.loadtxt('PCAMnist_train.csv', delimiter=',', skiprows=1)
   Y_train = data_train[:,-1] 
   X_train = data_train[:, :-1]  

   data_test= np.loadtxt('PCAMnist_test.csv', delimiter=',', skiprows=1)
   Y_test = data_test[:, -1]  
   X_test = data_test[:, :-1]  

   # data_train = np.loadtxt('mnist_train.csv', delimiter=',')
   # Y_train = data_train[:100,0]  # First column as target
   # X_train = data_train[:100,1:]  # Rest of the columns as feature vector

   # data_test = np.loadtxt('mnist_test.csv', delimiter=',')
   # Y_test = data_test[:,0]  # First column as target
   # X_test = data_test[:,1:]  # Rest of the columns as feature vector
   
   return X_train,X_test,Y_train,Y_test

def train_test_split():
   # Complete code for getting train test split
   return None  