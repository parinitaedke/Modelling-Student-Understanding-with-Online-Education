# TODO: complete this file.
from final_project.utils import *
from final_project.part_a.item_response import sigmoid, update_theta_beta
from final_project.part_a.matrix_factorization import als, update_u_z

import numpy as np


def bootstrap_data(data):
    """
    Returns a bootstrapped dataset the same size of data
    Args:
        data: dictionary of data

    Returns: a dictionary with resampled values

    """
    rand_indices_list = np.random.randint(len(data["is_correct"]), size=len(data["is_correct"]))

    sampled_data = {}

    for key in data.keys():
        sampled_data[key] = np.array(data[key])[rand_indices_list]

    return sampled_data


def item_response(train_data, lr, iterations):
    """
    Trains a 1-parameter IRT model
    Args:
        train_data: a dictionary with training data
        lr: learning rate
        iterations: number of iterations to train the model for

    Returns: the trained model parameters (theta and beta vectors)

    """
    resample_data = bootstrap_data(train_data)

    theta = np.random.random(max(resample_data['user_id']) + 1)
    theta = np.array(theta).reshape((len(theta), 1))
    beta = np.random.random(max(resample_data['question_id']) + 1)
    beta = np.array(beta).reshape((len(beta), 1))

    for i in range(iterations):
        theta, beta = update_theta_beta(resample_data, lr, theta, beta)

    return theta, beta


def bagged_predict(data, param_list):
    """
    Returns a bagged prediction of the data with the trained model parameters
    Args:
        data: a dictionary
        param_list: list of trained model parameters

    Returns: list of predictions

    """
    bagged_predictions = np.zeros(len(data['is_correct']))
    for model_param in param_list:
        bagged_predictions += predict(data, model_param[0], model_param[1])

    return bagged_predictions/len(param_list)


def predict(data, theta, beta):
    """Return vector of float prediction probabilities
    for the given data
    :param theta: Vector
    :param beta: Vector
    :return: Vector
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred


def ensemble_2(train_data, val_data, test_data):
    """
    Trains an IRT ensemble and produces training, validation and testing accuracies of the trained model.
    Args:
        train_data: a dictionary of training data
        val_data: a dictionary of validation data
        test_data: a dictionary of testing data

    Returns: None

    """

    # Setting hyperparameters
    lr = 0.01
    iters = 500
    num_models = 10
    models_list = []

    for i in range(num_models):
        print('Training Model ' + str(i + 1))
        base_model = item_response(train_data, lr, iters)
        models_list.append(base_model)

    # Get validation bagged predictions and accuracy
    train_bagged_preds = bagged_predict(train_data, models_list)
    train_accuracy = evaluate(train_data, train_bagged_preds)

    # Get validation bagged predictions and accuracy
    val_bagged_preds = bagged_predict(val_data, models_list)
    val_accuracy = evaluate(val_data, val_bagged_preds)

    # Get test bagged predictions and accuracy
    test_bagged_preds = bagged_predict(test_data, models_list)
    test_accuracy = evaluate(test_data, test_bagged_preds)

    print(f'Final Training Accuracy: {train_accuracy}')
    print(f'Final Validation Accuracy: {val_accuracy}')
    print(f'Final Test Accuracy: {test_accuracy}')


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    ensemble_2(train_data, val_data, test_data)


if __name__ == "__main__":
    main()
