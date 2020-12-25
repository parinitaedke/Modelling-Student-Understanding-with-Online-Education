"""
Part B of CSC311 Final Project
Proposed Modifications
"""

from final_project.utils import *
from final_project.part_a.item_response import sigmoid
from final_project.part_a.ensemble import item_response as one_param_item_response
from final_project.part_a.ensemble import bagged_predict, predict

import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns


def load_question_meta_data_csv(path):
    """
    A helper function to load the question_meta.csv file.
    Args:
        path: the path to the csv file

    Returns: a dictionary

    """
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))

                # this just makes the subject ids from the question_metadata from a string to an int array
                sid_list = []
                temp_sid = row[1].split('[')
                temp_sid = temp_sid[1].split(']')
                temp_sid = temp_sid[0].split(',')
                for i in range(len(temp_sid)):
                    sid_list.append(int(temp_sid[i]))

                data["subject_id"].append(sid_list)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_subject_meta_data_csv(path):
    """
    A helper function to load the subject_meta.csv file.
    Args:
        path: the path to the csv file

    Returns: a dictionary

    """
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "subject_id": [],
        "rank": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["subject_id"].append(int(row[0]))
                data["rank"].append(int(row[2]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def generate_k_values(question_metadata, subject_metadata):
    """
    Generates starting k_values for every question.
    Args:
        question_metadata: a dict with info from question_metadata.csv
        subject_metadata: a dict with info from temp.csv

    Returns: a dictionary of k-values

    """
    data = {}
    data["question_id"] = []
    data["subject_ranks"] = []
    for i in range(len(question_metadata["question_id"])):
        q_idx = question_metadata["question_id"][i]
        data["question_id"].append(q_idx)
        k_sum = 0.7
        for j in range(len(question_metadata["subject_id"][i])):
            sub = question_metadata["subject_id"][i][j]
            sub_rank = subject_metadata["rank"][sub]
            k_sum += sub_rank
        k_sum = k_sum / (len(question_metadata["subject_id"][i]) * 2)
        data["subject_ranks"].append(k_sum)
    return data


def bootstrap_data(data):
    """
    Returns a bootstraped sample of the size data
    Args:
        data: a dictionary

    Returns: a resampled dictionary

    """
    # print("Entered bootstrap_data")
    rand_indices_list = np.random.randint(len(data["is_correct"]), size=len(data["is_correct"]))

    sampled_data = {}
    sample_user_ids, sample_question_ids, sample_is_correct, = [], [], []

    for index in rand_indices_list:
        sample_user_ids.append(data["user_id"][index])
        sample_question_ids.append(data["question_id"][index])
        sample_is_correct.append(data["is_correct"][index])

    sampled_data["user_id"] = sample_user_ids
    sampled_data["question_id"] = sample_question_ids
    sampled_data["is_correct"] = sample_is_correct
    # sampled_data['ks'] = sample_ks

    return sampled_data


def two_param_update_theta_beta(data, lr, theta, beta, k_dict):
    """ Update theta and beta using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param k_dict: Dictionary
    :return: tuple of vectors
    """
    for i in range(len(data['is_correct'])):
        u = data['user_id'][i]
        q = data['question_id'][i]
        k_q_idx = k_dict["question_id"].index(q)
        k = k_dict['subject_ranks'][k_q_idx]
        c_ij = data["is_correct"][i]

        # Update theta
        x = theta[u]-beta[q]
        d_theta = (k*c_ij - (k*np.exp(k*x)*(1-c_ij)))/(1+np.exp(k*x))
        theta[u] += lr * d_theta

        # Update beta
        x = theta[u] - beta[q]
        d_beta = -(k*c_ij - (k*np.exp(k*x)*(1-c_ij)))/(1+np.exp(k*x))
        x = theta[u] - beta[q]
        beta[q] += lr * d_beta

        # Update k
        d_k = (c_ij*x - (1-c_ij)*(np.exp(k*x)*theta[u] - np.exp(k*x)*beta[q]))/(1+np.exp(k*x))
        k_dict['subject_ranks'][k_q_idx] += lr * d_k

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, k_dict


###########################################################################
# START                                                                  #
# This is for the negative log-likelihood part of our modified algorithm #
###########################################################################
def two_param_neg_log_likelihood(data, theta, beta, k_dict):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param k_dict: Dictionary
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
        # get the user of the ith entry
        u = data["user_id"][i]
        # get the question id of the ith entry
        q = data["question_id"][i]
        # get the c_ij of the ith entry
        c_ij = data["is_correct"][i]
        # get the index of where this question is located in the k_dict
        k_q_idx = k_dict["question_id"].index(q)
        # get the k_q_idx th subject rank
        k = k_dict['subject_ranks'][k_q_idx]
        log_lklihood += c_ij * np.log(sigmoid(k*(theta[u]-beta[q]))) \
                          + (1-c_ij)*np.log((1 - sigmoid(k*(theta[u]-beta[q]))))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def two_param_evaluate_irt(data, theta, beta, k_dict):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
   :param theta: Vector
   :param beta: Vector
   :param k_dict: Dictionary
   :return: float
   """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        k_q_idx = k_dict["question_id"].index(q)
        k = k_dict['subject_ranks'][k_q_idx]
        x = (k * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def two_param_irt(data, val_data, lr, iterations, k_dict):
    """ Train 2 parameter IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param k_dict: Dictionary
    :return: (theta, beta, val_acc_lst)
    """
    irt_predictions, val_acc_lst = [], []

    theta = np.random.random((len(np.unique(data['user_id'])), 1))  # 542 x 1 matrix
    beta = np.random.random((len(np.unique(data['question_id'])), 1))  # 1774 x 1 matrix

    for i in range(iterations):
        score_val = two_param_evaluate_irt(data=val_data, theta=theta, beta=beta, k_dict=k_dict)
        val_acc_lst.append(score_val)
        # theta, beta, k_list = two_param_update_theta_beta(resample_data, lr, theta, beta, k_list)
        theta, beta, k_dict = two_param_update_theta_beta(data, lr, theta, beta, k_dict)

    return theta, beta, k_dict, val_acc_lst


def main():
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    question_metadata = load_question_meta_data_csv("../data/question_meta.csv")
    subject_metadata = load_subject_meta_data_csv("../data/temp.csv")

    learning_rates = [0.001, 0.01, 0.03, 0.1, 0.5]
    iterations = [5, 10, 25, 50, 100]
    k_sum_list = generate_k_values(question_metadata, subject_metadata)
    validation_accuracies = []
    for lr in learning_rates:  # [0.03]:  # [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        for iter in iterations: # [120]:
            print("Testing model with {} learning rate and {} iterations...".format(lr, iter))
            theta, beta, k_list, val_acc_lst = two_param_irt(train_data, val_data, lr, iter, k_sum_list)
            validation_accuracies.append(val_acc_lst[-1])
            print("Validation accuracy: {}".format(val_acc_lst[-1]))
            # fig = plt.figure()
            # for i in [29, 88, 437, 777, 1001]:
            #     q_idx = k_sum_list["question_id"].index(i)
            #     k = k_sum_list["subject_ranks"][q_idx]
            #     x = k*(theta - beta[i])
            #     prob = sigmoid(x)
            #     plt.scatter(theta, prob, s=10.)
            # plt.title(
            #     'Probability each student has of correctly answering the 5 questions')
            # plt.ylabel('Probability of a correct response')
            # plt.xlabel('Theta')
            # plt.legend(['Question 1', 'Question 2', 'Question 3', 'Question 4',
            #             'Question 5'], loc='lower right')
            # plt.savefig('5_questions_Q2d_10_iters.png')

    sns.heatmap(np.array(validation_accuracies).reshape(5, 5), xticklabels=iterations,
                yticklabels=learning_rates, cmap="YlGnBu")
    plt.xlabel('Number of Iterations')
    plt.ylabel('Learning Rate (α)')
    plt.title('Validation Accuracy of 2-parameter IRT Model with Different Hyperparameters')
    plt.savefig('plots_b/2-param_IRT_hyperparameter_gridsearch.png')

###########################################################################
# END                                                                    #
# This is for the negative log-likelihood part of our modified algorithm #
###########################################################################


###########################################################################
# START                                                                  #
# This is for the ensemble part of our modified algorithm                #
###########################################################################
def two_param_item_response(train_data, test_data, lr, iterations, k_dict):
    """
    Trains a 2-parameter IRT model
    Args:
        train_data: dictionary of training data
        test_data: dictionary of testing data
        lr: learning rate
        iterations: number of iterations
        k_dict: dictionary of k_values

    Returns: trained model parameters (theta, beta and k_dict)

    """
    # resampled_data contains a bootstrapped sample of the original train data size. It also contains the respective
    # k values for the questions so that we can now index into resampled_data['ks][i] to get the i'th entries k-value.
    resampled_data = bootstrap_data(train_data)
    # resampled data will have a list of user_ids, question_ids, is_correct, and k_ranks

    theta = np.random.random((len(np.unique(resampled_data['user_id'])), 1))  # 542 x 1 matrix
    beta = np.random.random((len(np.unique(resampled_data['question_id'])), 1))  # 1774 x 1 matrix

    for i in range(iterations):
        # theta, beta, k_list = two_param_update_theta_beta(resample_data, lr, theta, beta, k_list)
        theta, beta, k_dict = two_param_update_theta_beta(resampled_data, lr, theta, beta, k_dict)

    return theta, beta, k_dict


def bagged_predict_2(data, param_list):
    """
     Returns a bagged prediction of the data with the trained model parameters
    Args:
        data: a dictionary
        param_list: list of trained model parameters

    Returns: list of predictions

    """
    bagged_predictions = np.zeros(len(data['is_correct']))
    for model_param in param_list:
        bagged_predictions += predict_2(data, model_param[0], model_param[1], model_param[2])
    return bagged_predictions/len(param_list)


def predict_2(data, theta, beta, k_dict):
    """Return vector of float prediction probabilities
    for the given data
    :param theta: Vector
    :param beta: Vector
    :return: Vector
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        k_q_idx = k_dict["question_id"].index(q)
        k = k_dict['subject_ranks'][k_q_idx]
        x = (k*(theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred


def ensemble_2(train_data, val_data, test_data, k_dict):
    """
    Trains a 2-parameter IRT ensemble and produces training, validation and testing accuracies of the trained model.
    Args:
        train_data: dictionary of training data
        val_data: dictionary of validation data
        test_data: dictionary of testing data
        k_dict: dictionary of k_values

    Returns: None

    """
    print("VAL data is_correct = {}".format(len(val_data["is_correct"])))

    num_models = 10
    lr = 0.03
    iters = 10
    two_param_list = []

    for i in range(num_models):
        print('Training Model ' + str(i + 1))
        model_params = two_param_item_response(train_data, val_data, lr, iters, k_dict)
        two_param_list.append(model_params)
    two_param_pred = bagged_predict_2(val_data, two_param_list)

    val_pred = two_param_pred
    evaluated_model = evaluate(val_data, val_pred)
    print("Bagged Validation accuracy: {}".format(evaluated_model))

    two_param_test_pred = bagged_predict_2(test_data, two_param_list)
    test_pred = two_param_test_pred
    evaluated_test_model = evaluate(test_data, test_pred)
    print("Bagged Test accuracy: {}".format(evaluated_test_model))

    two_param_train_pred = bagged_predict_2(train_data, two_param_list)
    train_pred = two_param_train_pred
    evaluated_train_model = evaluate(train_data, train_pred)
    print("Bagged Train accuracy: {}".format(evaluated_train_model))


def main_2():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    question_metadata = load_question_meta_data_csv("../data/question_meta.csv")
    subject_metadata = load_subject_meta_data_csv("subject_meta_modified.csv")

    k_sum_dict = generate_k_values(question_metadata, subject_metadata)
    ensemble_2(train_data, val_data, test_data, k_sum_dict)

###########################################################################
# END                                                                    #
# This is for the ensemble part of our modified algorithm                #
###########################################################################


if __name__ == "__main__":
    main_2()
