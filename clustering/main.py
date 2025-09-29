from model import KMeans, GMM
from utils import calc_NMI,train_test_split,read_data

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans as KMeans_from_library
from sklearn.mixture import GaussianMixture

def main() -> None:
    # set hyperparameters
    # X,y = read_data()
    # read data
    X_train,X_test,Y_train,Y_test = read_data(
        # X,y
        )
    # create a model
    gmm = GMM(k=10, max_iter=10)
    kmeans = KMeans(k=10, max_iter=10)

    # fit the model
    gmm.fit(X_train)
    kmeans.fit(X_train)

    # evaluate the models
    calc_NMI(gmm.predict(X_test),Y_test)
        
    kmeans_nmi = nmi(labels_true = Y_test, labels_pred = kmeans.predict(X_test))
    gmm_nmi = nmi(labels_true = Y_test, labels_pred = gmm.predict(X_test))
    calc_NMI(kmeans.predict(X_test),Y_test)

    print("From Implementation")
    print(f'Kmeans nmi={kmeans_nmi}, GMM nmi={gmm_nmi}\n')

    print("From Implementation (Using implemented NMI)")
    print(f'Kmeans nmi={calc_NMI(kmeans.predict(X_test),Y_test)}, GMM nmi={calc_NMI(gmm.predict(X_test),Y_test)}\n')

    
    kmeans_lib = KMeans_from_library(n_clusters=10, random_state=0)
    kmeans_lib.fit(X_train)
    kmeans_from_library = nmi(labels_true= Y_test, labels_pred= kmeans_lib.predict(X_test))
    gmm_from_library = 0

    
    gmm_lib = GaussianMixture(n_components=10, max_iter=10)
    gmm_lib.fit(X_train)
    gmm_from_library = nmi(labels_true = Y_test, labels_pred = gmm_lib.predict(X_test))
    
    print("From Library")
    print(f'Kmeans nmi={kmeans_from_library}, GMM nmi={gmm_from_library}\n')

    print("From Library (Using implemented NMI)")
    print(f'Kmeans nmi={calc_NMI(kmeans_lib.predict(X_test),Y_test)}, GMM nmi={calc_NMI(gmm_lib.predict(X_test),Y_test)}\n')
    


if __name__ == '__main__':
    main()
