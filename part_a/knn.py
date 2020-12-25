from sklearn.impute import KNNImputer
from final_project.utils import *

import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)

    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)  # 542 x 1774 matrix

    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy for IMPUTE_BY_USER: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)

    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)  # 1774 x 542 matrix

    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy for IMPUTE_BY_ITEM: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()  # 542 x 1774 matrix
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    k_list = [1, 6, 11, 16, 21, 26]
    k_user_accuracies = []
    k_item_accuracies = []

    # runs kNN for different values of k_list
    for k in k_list:
        print(k)
        k_user_accuracies.append(knn_impute_by_user(sparse_matrix, val_data, k))
        k_item_accuracies.append(knn_impute_by_item(sparse_matrix,val_data,k))

    print(k_user_accuracies)
    print(k_item_accuracies)

    # choose k* by getting the index of maximum accuracy in the list
    best_user_k_index = k_user_accuracies.index(max(k_user_accuracies))
    best_item_k_index = k_item_accuracies.index(max(k_item_accuracies))

    # compute final test accuracy for impute_by_user and impute_by_item
    test_user_accuracy = knn_impute_by_user(sparse_matrix, test_data, best_user_k_index)
    test_item_accuracy = knn_impute_by_item(sparse_matrix, test_data, best_item_k_index)

    print("Impute_by_user: Test Accuracy: {}".format(test_user_accuracy))
    print("Impute_by_item: Test Accuracy: {}".format(test_item_accuracy))

    # plot the accuracy on the validation data as a function of k
    # plt.plot(k_list, k_user_accuracies, label='Impute_by_user: Validation accuracy')
    plt.plot(k_list, k_item_accuracies, label='Impute_by_item: Validation accuracy')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend()

    print("Plotting...")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
