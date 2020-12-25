from final_project.utils import *
from scipy.linalg import sqrtm

import numpy as np

import matplotlib.pyplot as plt


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    # compute mean for each question and fill in the nan with the mean
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0) # gives us a vector of 1774 means
    # repeating the item_means vector for 542 times to form 542 X 1774 matrix
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k] # top left k x k matrix of the s diagonal matrix
    Q = Q[:, 0:k] # taking first k columns as each column is an eigenvectors
    Ut = Ut[0:k, :] # taking first k rows as each row is an eigenvector
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    u[n] += lr * np.dot((c - np.dot(u[n].T, z[q])), z[q])
    z[q] += lr * np.dot((c - np.dot(u[n].T, z[q])), u[n])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, val_data):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))  # 542 x k matrix
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))  # 1774 x k matrix

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_squared_error_losses, val_squared_error_losses = [], []
    for iteration in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        print("iter: {}".format(iteration))
        train_squared_error_losses.append(squared_error_loss(train_data, u, z))
        val_squared_error_losses.append(squared_error_loss(val_data, u, z))

    mat = np.dot(u, z.T)  # 542 x 1774 matrix

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_squared_error_losses, val_squared_error_losses


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    k_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    for k in k_list:
        new_mtx = svd_reconstruct(train_matrix, k)
        score = sparse_matrix_evaluate(val_data, new_mtx)
        print(k)
        print(score)
    new_mtx = svd_reconstruct(train_matrix, 25)
    val_acc = sparse_matrix_evaluate(val_data, new_mtx)
    tst_acc = sparse_matrix_evaluate(test_data, new_mtx)
    print("Validation Accuracy: {}".format(val_acc))
    print("Test Accuracy: {}".format(tst_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # learning rate, k, num_iterations
    # k = 40, learning rate = 0.05, num_iterations = 80000*, 128000
    k_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    k_list = [40]
    learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
    iterations_list = [80000, 128000, 150000]

    val_accuracies = np.ones((len(k_list), len(learning_rate_list), len(iterations_list)))
    temp_accuracies = []

    training = False

    if training:
        # select the best k∗ that achieves the highest validation accuracy.
        for k in range(len(k_list)):
            print("K: {}".format(k))
            for lr in range(len(learning_rate_list)):  # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
                print("learning rate: {}".format(lr))
                for iteration in range(len(iterations_list)):  # [5, 10, 20, 30, 50]
                    print("iteration: {}".format(iteration))
                    mat, train_SEL, val_SEL = als(train_data, k_list[k], learning_rate_list[lr], iterations_list[iteration], val_data)
                    val_acc = sparse_matrix_evaluate(val_data, mat)
                    temp_accuracies.append(val_acc)
                    val_accuracies[k][lr][iteration] = val_acc

        print(val_accuracies)
        print(temp_accuracies)
        print(val_accuracies)
        print(max(temp_accuracies))
        print(temp_accuracies.index(max(temp_accuracies)))

    # Q2e.
    mat, train_SEL, val_SEL = als(train_data, 40, 0.05, 128000, val_data)
    print("Validation Accuracy: {}".format(sparse_matrix_evaluate(val_data, mat)))
    print("Test Accuracy:{}".format(sparse_matrix_evaluate(test_data, mat)))
    iteration_labels = [i for i in range(128000)]
    fig = plt.figure()
    plt.plot(iteration_labels, train_SEL, label='Train SEL')
    plt.plot(iteration_labels, val_SEL, label='Validation SEL')
    plt.xlabel('iterations')
    plt.ylabel('squared error loss')
    plt.legend()
    plt.savefig("plots/MatrixFactorization.png")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
