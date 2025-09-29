from utils import get_data
from model import SVM_class
from typing import Tuple


def get_hyperparameters() -> Tuple[float, int, float, float]:
    #implement this
    # get the hyperparameters
    # learning_rate = float(input("Enter the learning rate: "))
    # num_iters = int(input("Enter the number of iterations: "))
    # C = float(input("Enter the value of C: "))
    learning_rate = 0.0001
    num_iters = 1000
    C = 0.1
    batch_fraction = 1.0/600.0
    return learning_rate, num_iters, C, batch_fraction
    raise NotImplementedError


def main() -> None:
    # hyperparameters
    learning_rate, num_iters, C, batch_fraction = get_hyperparameters()

    # get data
    X_train, X_test, y_train, y_test = get_data()

   
       
    # create a model
    svm = SVM_class()

    # fit the model
    svm.fit(
            X_train, y_train, C=C,
            learning_rate=learning_rate,
            num_iters=num_iters,
            batch_fraction=batch_fraction,
        )

    # evaluate the model
    accuracy = svm.accuracy_score(X_test, y_test)
    precision = svm.precision_score(X_test, y_test)
    recall = svm.recall_score(X_test, y_test)
    f1_score = svm.f1_score(X_test, y_test)



    print(f'accuracy={accuracy}, precision={precision}, recall={recall}, f1_score={f1_score}')
    # print(f'{accuracy} {precision} {recall} {f1_score}')


if __name__ == '__main__':
    main()
