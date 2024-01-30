from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import time
import pandas as pd
from time import sleep
from torch import nn
from hyperopt import hp, fmin, tpe, STATUS_OK, space_eval

def get_pred_sir_original(dist_train_cobra_original, dist_test_cobra_original, y_train_l):
    start_time = time.time()
    def gradient(f,dist,y ,x0, eps):
        return np.array([(f(dist,y,x0+eps) - f(dist,y,x0-eps))/(2*eps)])
    precision = 10 ** (-5)
    epsilon = 10 ** (-10)
    minimum = np.min(dist_train_cobra_original)
    maximum = np.max(dist_train_cobra_original)
    bws = np.linspace(minimum, maximum, num = 10)
    
    initial_tries = [kappa_cross_validation_error_sir(dist_train_cobra_original,y_train_l,b) for b in bws]
    bw0 = bws[np.argmin(initial_tries)]
    true_bws = bw0
    grad = gradient(kappa_cross_validation_error_sir,dist_train_cobra_original,y_train_l, bw0, precision)
    r0 = 0.01 / abs(grad)        # make the first step exactly equal to `learning-rate`.
    rate = lambda x, y: x*y          # the learning rate can be varied, and speed defines this change in learning rate.
    
    count = 0
    grad0 = grad
    while count < 100:
        bw = bw0 - rate(count, r0) * grad
        if bw < 0 or np.isnan(bw):
            bw = bw0 * 0.5
        if count > 3:
            if np.sign(grad)*np.sign(grad0) < 0:
                r0 = r0 * 0.8
            if test_threshold > epsilon:
                bw0, grad0 = bw, grad
            else:
                break
        relative = abs((bw - bw0) / bw0)
        test_threshold = (np.mean(relative)+ np.mean(abs(grad)))/2
        grad = gradient(kappa_cross_validation_error_sir,dist_train_cobra_original,y_train_l, bw0, precision)
        count += 1
    try:
        opt_bw = bw[0]
    except:
        opt_bw = bw
    end_time = time.time()
    time_taken_in_search = end_time - start_time
    def proposed_kernel(distance_matrix, epsilon, beta = 200):
        print(f"distance_matrix shape during testin {distance_matrix.shape}")
        return np.exp(beta*epsilon)/(np.exp(beta*epsilon) + np.sum(np.exp(beta*distance_matrix), axis =-1))
    start_time = time.time()
    D_k = proposed_kernel(dist_test_cobra_original, opt_bw)
    print(f"D_k shape while testing {D_k.shape}")
    if  np.any(np.isnan(D_k)):
        print(f"Warning : opt_bw is very large {opt_bw} and {D_k} setting bw to {true_bws}")
        opt_bw = true_bws
        D_k = proposed_kernel(dist_test_cobra_original, opt_bw)


    
    D_k_ = np.sum(D_k, axis=1, dtype=np.float64)
    print(f"D_k_ shape while testing {D_k_.shape}")
    D_k_[D_k_ == 0] = np.Inf
    #print(f"Dist test shape {dist_test_cobra_original.shape}")
    #print(f"Optimal Bandwidth {opt_bw}, D_k_ shape {D_k_.shape} D_k shape {D_k.shape} , y_train_l shape {y_train_l.shape}")
    res = np.sum(D_k*y_train_l,axis=-1)
    #print(f"Res shape {res.shape}")
    np.divide(res, D_k_, out = res, where = D_k_ != 0)
    res[res == 0] = res[res != 0].mean()
    end_time = time.time()
    time_taken_in_prediction = end_time - start_time
    res2 = cobra_predict(dist_test_cobra_original, y_train_l, opt_bw , alpha = 1)
    
    return res , res2 , time_taken_in_search , time_taken_in_prediction

def kappa_cross_validation_error_sir(distance_matrix, y_l_,epsilon = 1):
    ids = pd.DataFrame({'shuffle': list(range(5))})
    index_shuffled = ids.sample(distance_matrix.shape[0], replace=True, random_state=23).shuffle.values
    #print(f"Epsilon : {epsilon}")
    cost = np.full(5, fill_value = np.float64)
    def proposed_kernel(distance_matrix, epsilon, beta = 100):
        print(f"distance_matrix shape during training {distance_matrix.shape}")
        return np.exp(beta*epsilon)/(np.exp(beta*epsilon) + np.sum(np.exp(beta*distance_matrix), axis =-1))
    for i in range(5):
        D_k = proposed_kernel(distance_matrix[index_shuffled != i,:][:,index_shuffled == i],epsilon)
        print(f"D_k shape during training {D_k.shape}")
        D_k_ = np.sum(D_k, axis=0, dtype=np.float64)
        print(f"D_k_ shape during training {D_k_.shape}")
        D_k_[D_k_ == 0] = np.Inf
        res = np.matmul(y_l_[index_shuffled != i], D_k)/D_k_
        #print(f"Optimal Bandwidth {epsilon}, D_k_ shape {D_k_.shape} D_k shape {D_k.shape} , y_train_l shape {y_l_[index_shuffled != i].shape}")
        try :
            cost[i] = mean_squared_error(res, y_l_[index_shuffled == i])
        except:
            res = np.ones(len(y_l_[index_shuffled == i]))*np.mean(y_l_[index_shuffled != i])
            cost[i] = mean_squared_error(res, y_l_[index_shuffled == i])
            print(f"Error {res} , {y_l_[index_shuffled == i]}")
            print(f"D_k_ {D_k_} and epsilon {epsilon}")


    cost_ = cost.mean()
    return cost_

def cobra_predict(distance_matrix, y_l_, bandwidth, alpha = 1):
    indicator = np.sum(distance_matrix <= bandwidth,-1) >= distance_matrix.shape[-1]
    y_pred = np.ones((distance_matrix.shape[0]))*np.mean(y_l_)
    numerator = np.sum(indicator*y_l_, axis = 1)
    denonminator = np.sum(indicator, axis = 1)
    np.divide(numerator, denonminator, out = y_pred, where = denonminator != 0)
    return y_pred

def get_pred_gradient_original(dist_train_cobra_original, dist_test_cobra_original, y_train_l):
    dist_train_cobra_original = np.mean(dist_train_cobra_original, axis=2)
    dist_test_cobra_original = np.mean(dist_test_cobra_original, axis=2)
    start_time = time.time()
    distance_matrix = {}
    index_each_fold = {}
    precision = 10 ** (-5)
    epsilon = 10 ** (-10)
    bws = np.linspace(0.0001, 1/np.var(y_train_l), num = 10)
    initial_tries = [kappa_cross_validation_error(dist_train_cobra_original,y_train_l,bandwidth=b) for b in bws]
    bw0 = bws[np.argmin(initial_tries)]
    true_bws = bw0
    def gradient(f,dist,y ,x0, eps = precision):
        return np.array([(f(dist,y,x0+eps) - f(dist,y,x0-eps))/(2*eps)])
    grad = gradient(kappa_cross_validation_error,dist_train_cobra_original,y_train_l, bw0, precision)
    r0 = 0.01 / abs(grad)        # make the first step exactly equal to `learning-rate`.
    rate = lambda x, y: x*y          # the learning rate can be varied, and speed defines this change in learning rate.
    
    count = 0
    grad0 = grad
    while count < 100:
        bw = bw0 - rate(count, r0) * grad
        if bw < 0 or np.isnan(bw):
            bw = bw0 * 0.5
        if count > 3:
            if np.sign(grad)*np.sign(grad0) < 0:
                r0 = r0 * 0.8
            if test_threshold > epsilon:
                bw0, grad0 = bw, grad
            else:
                break
        relative = abs((bw - bw0) / bw0)
        test_threshold = np.mean([relative, abs(grad)])
        grad = gradient(kappa_cross_validation_error,dist_train_cobra_original,y_train_l, bw0, precision)
        count += 1
    try:
        opt_bw = bw[0]
    except:
        opt_bw = bw
    end_time = time.time()
    time_taken_in_search = end_time - start_time
    gaussian = lambda x,y: np.exp(-x*y)
    start_time = time.time()
    D_k = gaussian(dist_test_cobra_original, opt_bw)
    if  np.any(np.isnan(D_k)):
        print(f"Warning : opt_bw is very large {opt_bw} and {D_k} setting bw to {true_bws}")
        opt_bw = true_bws
        D_k = gaussian(dist_test_cobra_original, opt_bw)
    D_k_ = np.sum(D_k, axis=1, dtype=np.float64)
    D_k_[D_k_ == 0] = np.Inf
    #print(f"Optimal Bandwidth {opt_bw}, D_k_ shape {D_k_.shape} D_k shape {D_k.shape} , y_train_l shape {y_train_l.shape}")
    res = np.sum(D_k*y_train_l , axis = -1)/D_k_
    if not np.all(res == 0):
        res[res == 0] = res[res != 0].mean()
    else:
        res = np.ones_like(res)*np.mean(y_train_l)

    end_time = time.time()
    time_taken_in_prediction = end_time - start_time
    return res , res , time_taken_in_search , time_taken_in_prediction


def kappa_cross_validation_error(distance_matrix, y_l_,bandwidth = 1):
    ids = pd.DataFrame({'shuffle': list(range(5))})
    index_shuffled = ids.sample(distance_matrix.shape[0], replace=True, random_state=23).shuffle.values

    cost = np.full(5, fill_value = np.float64)
    gaussian = lambda x,y: np.exp(-x*y)
    for i in range(5):
        D_k = gaussian(distance_matrix[index_shuffled != i,:][:,index_shuffled == i], bandwidth)
        D_k_ = np.sum(D_k, axis=0, dtype=np.float64)
        D_k_[D_k_ == 0] = np.Inf
        res = np.matmul(y_l_[index_shuffled != i], D_k)/D_k_
        try :
            cost[i] = mean_squared_error(res, y_l_[index_shuffled == i])
        except:
            print(f"Error {res} , {y_l_[index_shuffled == i]}")
            print(f"D_k_ {D_k_}")
    cost_ = cost.mean()
    return cost_



def get_pred_cobra_original(dist_train_cobra_original, dist_test_cobra_original, y_train_l):
    n_grid = 100
    minimum = np.min(dist_train_cobra_original)
    maximum = np.max(dist_train_cobra_original)
    bws = np.linspace(minimum, maximum, num = n_grid)
    mse = np.inf
    best_bw = 0
    start_time = time.time()
    for bw in bws:
        cost = kappa_cross_validation_error_grid(dist_train_cobra_original,y_train_l,bandwidth=bw)
        if cost < mse:
            mse = cost
            best_bw = bw
    end_time = time.time()
    time_taken_in_search = end_time - start_time
    #print(f"Best Bandwidth {best_bw} , Best MSE {mse}")
    start_time = time.time()
    y_pred = cobra_predict(dist_test_cobra_original, y_train_l, best_bw)
    end_time = time.time()
    return y_pred, y_pred, time_taken_in_search, end_time - start_time

def kappa_cross_validation_error_grid(distance_matrix, y_l_,bandwidth = 1):
    ids = pd.DataFrame({'shuffle': list(range(5))})
    index_shuffled = ids.sample(distance_matrix.shape[0], replace=True, random_state=23).shuffle.values

    cost = np.full(5, fill_value = np.float64)
    for i in range(5):
        D_k = distance_matrix[index_shuffled != i,:][:,index_shuffled == i]
        D_k = D_k.transpose(1,0,2)
        corresponding_y = y_l_[index_shuffled != i]
        indicator = np.sum(D_k <= bandwidth,axis=-1) >= D_k.shape[-1]
        res = np.ones(len(y_l_[index_shuffled == i]))*np.mean(corresponding_y)
        numerator = np.sum(indicator*corresponding_y, axis = 1)
        
        np.divide(numerator, np.sum(indicator, axis = 1), out = res, where = np.sum(indicator, axis = 1) != 0)
        cost[i] = mean_squared_error(res, y_l_[index_shuffled == i])
    cost_ = cost.mean()
    return cost_


def get_pred_random(dist_train_cobra_original, dist_test_cobra_original, y_train_l):
    n_grid = 100
    minimum = np.min(dist_train_cobra_original)
    maximum = np.max(dist_train_cobra_original)
    mse = np.inf
    best_bw = 0
    start_time = time.time()
    for bw in range(n_grid):
        # get random bandwidth from the range of minimum and maximum
        bw = np.random.uniform(minimum, maximum)
        cost = kappa_cross_validation_error_grid(dist_train_cobra_original,y_train_l,bandwidth=bw)
        if cost < mse:
            mse = cost
            best_bw = bw
    end_time = time.time()
    time_taken_in_search = end_time - start_time
    #print(f"Best Bandwidth {best_bw} , Best MSE {mse}")
    start_time = time.time()
    y_pred = cobra_predict(dist_test_cobra_original, y_train_l, best_bw)
    end_time = time.time()
    return y_pred, y_pred, time_taken_in_search, end_time - start_time
