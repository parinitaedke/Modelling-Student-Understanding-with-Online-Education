from final_project.utils import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # print("Entering neg_log_likelihood")
    log_lklihood = 0.

    for i in range(len(data['is_correct'])):
        # looping over each entry
        u = data["user_id"][i]
        q = data["question_id"][i]
        c_ij = data["is_correct"][i]
        log_lklihood += c_ij * np.log(sigmoid(theta[u]-beta[q])) \
                          + (1-c_ij)*np.log(1-sigmoid(theta[u]-beta[q]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Update theta
    thetas = np.array(theta[data['user_id']])  # 56688 x 1 matrix
    betas = np.array(beta[data['question_id']])  # 56688 x 1 matrix

    # Compute sigmoid of each theta_i, beta_j pair
    likelihoods_array = np.array(sigmoid(thetas - betas))  # 56688 x 1 matrix

    # Subtract likelihoods_array from is_correct
    is_correct = np.array(data['is_correct']).reshape((likelihoods_array.shape[0], likelihoods_array.shape[1]))
    partials = np.array(is_correct - likelihoods_array)  # 56688 x 1 matrix
    partials = partials.reshape((partials.shape[0],))

    # Group by i (user_id) and sum
    bins = np.array(np.bincount(data['user_id'], weights=partials))
    theta = theta + lr * bins.reshape((theta.shape[0], theta.shape[1]))

    # **************************************

    # Update beta using new theta
    thetas = np.array(theta[data['user_id']])
    betas = np.array(beta[data['question_id']])

    # Compute sigmoid of each theta_i, beta_j pair
    likelihoods_array = np.array(sigmoid(thetas - betas))

    # Subtract is_correct from likelihoods array
    is_correct = np.array(data['is_correct']).reshape((likelihoods_array.shape[0], likelihoods_array.shape[1]))
    partials = np.array(likelihoods_array - is_correct)  # 56688 x 1 matrix
    partials = partials.reshape((partials.shape[0],))

    # Group by j (question_id) and sum
    bins = np.array(np.bincount(data['question_id'], weights=partials))
    beta = beta + lr * bins.reshape((beta.shape[0], beta.shape[1]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros((len(np.unique(data['user_id'])), 1))  # 542 x 1 matrix
    beta = np.zeros((len(np.unique(data['question_id'])), 1))  # 1774 x 1 matrix

    val_acc_lst = []
    val_likelihood = []
    train_likelihood = []

    for i in range(iterations):
        # print(i)
        neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_likelihood.append(neg_lld)
        score_val = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score_val)
        print("NLLK: {} \t Score: {}".format(neg_lld, score_val))

        neg_lld_t = neg_log_likelihood(data, theta=theta, beta=beta)
        train_likelihood.append(neg_lld_t)

        # print("Update theta beta")
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, val_likelihood, train_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # learning rate, num_iterations, how to initialize theta and beta
    training = False
    optimal_lr = -1
    optimal_iters = -1

    if training:
        learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        iterations = [10, 50, 100, 250, 500, 1000]
        validation_accuracies = []

        for lr in learning_rates:  # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            for iter in iterations:  # [10, 50, 100, 250, 500, 1000]
                print("Testing model with {} learning rate and {} iterations...".format(lr, iter))
                theta, beta, val_acc_lst, val_lh, train_lh = irt(train_data, val_data, lr, iter)
                validation_accuracies.append(val_acc_lst[-1])

        # Plot accuracy heatmap
        sns.heatmap(np.array(validation_accuracies).reshape(6, 6), xticklabels=iterations,
                              yticklabels=learning_rates, cmap="YlGnBu")
        plt.xlabel('Number of Iterations')
        plt.ylabel('Learning Rate (α)')
        plt.title('Validation Accuracy of IRT Model with Different Hyperparameter')
        plt.savefig('plots/IRT_hyperparameter_gridsearch.png')
        plt.show()

        # Choose optimal hyperparameters
        optimal_lr = learning_rates[np.argmax(validation_accuracies) // 5]
        optimal_iters = iterations[np.argmax(validation_accuracies) % 5]
        print("Optimal hyperparameters: {} learning rate with {} iterations".format(optimal_lr, optimal_iters))

    else:
        optimal_lr = 0.01
        optimal_iters = 10
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)
    print('Training optimal model...')
    theta, beta, val_acc_lst, val_lh, train_lh = irt(train_data, val_data, optimal_lr, optimal_iters)

    # Plotting training curve that shows the training and validation log-likelihoods as a function of iteration.
    fig = plt.figure()
    plt.plot(list(range(optimal_iters)), train_lh,  label="training")
    plt.plot(list(range(optimal_iters)), val_lh, label="validation")
    plt.title('Negative log-likelihood')
    plt.ylabel('Negative log-likelihood')
    plt.xlabel('Number of Iteration')
    plt.legend()
    plt.savefig("plots/Negative_log_likelihood_lr{}_iter{}.png".format(optimal_lr, optimal_iters))

    # Getting final training, validation and testing accuracies
    final_train_acc = evaluate(train_data, theta, beta)
    final_val_acc = evaluate(val_data, theta, beta)
    final_test_acc = evaluate(test_data, theta, beta)

    print("Final Training Accuracy:{}".format(final_train_acc))
    print("Final Validation Accuracy: {}".format(final_val_acc))
    print("Final Test Accuracy: {}".format(final_test_acc))
    #####################################################################

    # Q2.d
    # choose five random questions
    fig = plt.figure()
    for i in [29, 88, 437, 777, 1001]:
        x = theta - beta[i]
        prob = sigmoid(x)
        plt.scatter(theta, prob, s=10.)
    plt.title('Probability each student has of correctly answering the 5 questions')
    plt.ylabel('Probability of a correct response')
    plt.xlabel('Theta')
    plt.legend(['Question 1', 'Question 2', 'Question 3', 'Question 4', 'Question 5'], loc='lower right')
    plt.savefig('plots/5_questions_Q2d_10_iters.png')
    # plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
