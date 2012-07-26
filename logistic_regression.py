# -*-coding:utf-8-*-
# 

#implement logistic regression with out library such as numpy

from math import e, sqrt, log

def load_data(file_path, delimiter=','):
    ''' load data from file '''
    X = [line.strip().split(delimiter)[:-1] for line in open(file_path)]
    y = [int(line.strip().split(delimiter)[-1]) for line in open(file_path)]
    return X, y

def feature_normalize(X):
    m = len(X)
    feature_num = len(X[0])
    column_mean = [sum([float(X[j][i]) for j in xrange(m)]) / m for i in xrange(feature_num)]
    print column_mean
    column_std = [sqrt(sum([float(X[j][i]) ** 2 for j in xrange(m)]) / m)  for i in xrange(feature_num)]
    print column_std
    return [[(float(X[i][j]) - column_mean[j]) / column_std[j] for j in xrange(feature_num)] for i in xrange(m)]

def sigmoid_function(x):
    '''Compute the sigmoid funtion '''
    return 1.0 / (1.0 + e ** (-1.0 * x))

def cost_function(X, y, theta):
    ''' compute cost function '''
    m = len(y)
    num_of_feature  = len(theta)
    h = [sigmoid_function(sum([X[i][j] * theta[j] for j in xrange(num_of_feature)])) for i in xrange(m)]
    log_h = [log(x) for x in h]
    log_1_minus_h = [log(1.0 - x) for x in h]
    return (-1.0 / m) * (sum([log_h[i] * y[i] for i in xrange(m)]) + sum([(1.0 - y[i]) * log_1_minus_h[i] for i in xrange(m)]))

def gradient_decent(X, y, theta, alpha, num_of_iters):
    m = len(y)
    for i in xrange(num_of_iters):
        tmp_theta = []
        for j in xrange(len(theta)):
            sum_predict = 0.0
            for k in xrange(m):
                sum_predict += (sigmoid_function(sum([X[k][a] * theta[a] for a in xrange(feature_num)])) - y[k]) * X[k][j]
            tmp_theta.append(theta[j] - alpha * sum_predict / m)
        theta = tmp_theta
        print theta
        print cost_function(X, y, theta)
    return theta

def predict(test_data_path, theta, delimiter=','):
    X = [[1] + line.strip().split(delimiter)[:-1] for line in open(test_data_path)]
    y = [int(line.strip().split(delimiter)[-1]) for line in open(test_data_path)]
    X = feature_normalize(X)
    idx = 0
    for x in X:    
        h = sigmoid_function(sum([x[i] * theta[i] for i in xrange(len(x))]))
        r = 0
        if h >= 0.5:
            r = 1
        print r, y[idx]
        idx += 1

X, y = load_data('train.data')

X = feature_normalize(X)

m = len(y)
feature_num = len(X[0])

X = [[1] + X[i] for i in xrange(m)]

print X

theta = [1 for i in xrange(feature_num + 1)]
alpha = 0.1
num_of_iters = 600

theta = gradient_decent(X, y, theta, alpha, num_of_iters)

predict('test.data', theta)
