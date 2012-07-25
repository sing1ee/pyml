# -*-coding:utf-8-*-
# 

#implement logistic regression with out library such as numpy

from math import e, sqrt, log

def load_data(file_path, delimiter=','):
    ''' load data from file '''
    X = [line.strip().split(delimiter)[:-1] for line in open(file_path)]
    y = [line.strip().split(delimiter)[-1] for line in open(file_path)]
    return X, y

def feature_normalize(X):
    m = len(X)
    feature_num = len(X[0])
    column_mean = [sum([X[i][j] for j in xrange(m)]) / m for i in xrange(feature_num)]
    column_std = [sqrt(sum([X[i][j] ** 2 for j in xrange(m)]) / m)  for i in xrange(feature_num)]
    return [[(X[i][j] - column_mean[j]) / column_std[j] for i in xrange(m)] for j in xrange(feature_num)]

def sigmoid_function(x):
    '''Compute the sigmoid funtion '''
    return 1.0 / (1.0 + e ** (-1.0 * x))

def cost_function(X, y, theta):
    ''' compute cost function '''
    m = len(y)
    h = [sigmoid(sum([X[i][j] * y[j] for j in xrange(m)])) for i in xrange(len(X))]
    log_h = [log(x) for x in h]
    log_1_minus_h = [log(1.0 - x) for x in h]
    return (-1.0 / m) * (sum([log_h[i] * y[i] for i in xrange(m)] + sum([(1.0 - y[i]) * log_1_minus_h[i] for i in xrange(m)])))

def gradient_decent(X, y, theta, alpha, num_of_iters):
    m = len(y)
    for i in xrange(num_of_iters):
        tmp_theta = []
        for j in xrange(len(theta)):
            sum_predict = 0.0
            for k in xrange(m):
                sum_predict += (sum([sigmoid(X[k][a] * y[a]) for a in xrange(m)]) - y[k]) * X[k][j]
            tmp_theta[j] = theta[j] - alpha * sum_predict / m
        theta = tmp_theta
        print cost_function(X, y, theta):
    return theta

def predict(x, theta):
    h = sigmoid_function(sum([x[i] * theta[i] for i in xrange(len(x))]))
    if h >= 0.5:
        return True
    return False


