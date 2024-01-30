from uci_datasets import all_datasets
from uci_datasets import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import time
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from idea_util import get_pred_sir_original, get_pred_cobra_original, get_pred_gradient_original
import csv
# do not raise warnings numpy
np.seterr(all='ignore')

writer = csv.writer(open("comparison_df2.csv", 'w'))

data_names = ["concreteslump", "energy" ,"stock" ,"yacht" , "airfoil" , "autos" , "breastcancer" ,"concrete","gas",  "machine" ,"pendulum" , "servo", "sml" ,"wine" ]
models = [LassoCV(), RidgeCV(), DecisionTreeRegressor(), RandomForestRegressor() , SVR()]
n_grid = 100
n_iter = 100

print(f"Starting Analysis for {data_names} datasets")
result_df = pd.DataFrame(columns=['dataset', 'fold', 'model', 'mse', 'R2' , 'data_trained_on', 'data_tested_on', 'time_taken_fit', 'time_taken_infer'])

data_index = 1
times_one_out_better = 0
times_val_better = 0
times_cv_better = 0
total_experiments = 0
for name in data_names:
    data = Dataset(name)
    print(f"Starting Analysis for {name} , {data_index}/{len(data_names)}")
    data_index = data_index + 1
    data_seed = data_index

   
    
    for fold_idx in range(10):
        fold_seed = fold_idx
        np.random.seed(data_seed+fold_seed)
        print(f"Starting Fold {fold_idx}")
        x_train, y_train, x_test, y_test = data.get_split(split=fold_idx)
        shuffled_index = np.random.permutation(len(x_train))
        x_train = x_train[shuffled_index]
        y_train = y_train[shuffled_index]

        # scale the dataset
        scaler_whole = StandardScaler()
        x_train_transformed = scaler_whole.fit_transform(x_train)
        x_test_transformed_whole = scaler_whole.transform(x_test)
        scaler_whole_y = StandardScaler()
        y_train_transformed = scaler_whole_y.fit_transform(y_train.reshape(-1, 1))
        y_test_transformed = scaler_whole_y.transform(y_test.reshape(-1, 1))

        
        
        dk = x_train[:len(x_train)//2]
        dl = x_train[len(x_train)//2:]
        y_train_k = y_train[:len(y_train)//2]
        y_train_l = y_train[len(y_train)//2:]

        scaler_dk = StandardScaler()
        dk_transformed =  scaler_dk.fit_transform(dk)
        dl_transformed = scaler_dk.transform(dl)
        x_test_transformed_dk = scaler_dk.transform(x_test)
        x_train_transformed_dk = scaler_dk.transform(x_train)
        scaler_dk_y = StandardScaler()
        y_train_k_transformed = scaler_dk_y.fit_transform(y_train_k.reshape(-1, 1))
        y_train_l_transformed = scaler_dk_y.transform(y_train_l.reshape(-1, 1))
        y_test_transformed_dk = scaler_dk_y.transform(y_test.reshape(-1, 1))

        # seve all variables in npy files
        # np.save(f"dk_{name}_{fold_idx}.npy", dk)
        # np.save(f"dl_{name}_{fold_idx}.npy", dl)
        # np.save(f"y_train_k_{name}_{fold_idx}.npy", y_train_k)
        # np.save(f"y_train_l_{name}_{fold_idx}.npy", y_train_l)
        # np.save(f"x_test_{name}_{fold_idx}.npy", x_test)
        # np.save(f"y_test_{name}_{fold_idx}.npy", y_test)
        # np.save(f"x_train_{name}_{fold_idx}.npy", x_train)
        # np.save(f"y_train_{name}_{fold_idx}.npy", y_train)
        # np.save(f"x_train_transformed_{name}_{fold_idx}.npy", x_train_transformed)
        # np.save(f"x_test_transformed_whole_{name}_{fold_idx}.npy", x_test_transformed_whole)
        # np.save(f"y_train_transformed_{name}_{fold_idx}.npy", y_train_transformed)
        # np.save(f"y_test_transformed_{name}_{fold_idx}.npy", y_test_transformed)
        # np.save(f"dk_transformed_{name}_{fold_idx}.npy", dk_transformed)
        # np.save(f"dl_transformed_{name}_{fold_idx}.npy", dl_transformed)
        # np.save(f"x_test_transformed_dk_{name}_{fold_idx}.npy", x_test_transformed_dk)
        # np.save(f"x_train_transformed_dk_{name}_{fold_idx}.npy", x_train_transformed_dk)
        # np.save(f"y_train_k_transformed_{name}_{fold_idx}.npy", y_train_k_transformed)
        # np.save(f"y_train_l_transformed_{name}_{fold_idx}.npy", y_train_l_transformed)
        # np.save(f"y_test_transformed_dk_{name}_{fold_idx}.npy", y_test_transformed_dk)
        # np.save(f"scaler_whole_{name}_{fold_idx}.npy", scaler_whole)
        # np.save(f"scaler_whole_y_{name}_{fold_idx}.npy", scaler_whole_y)
        # np.save(f"scaler_dk_{name}_{fold_idx}.npy", scaler_dk)
        # np.save(f"scaler_dk_y_{name}_{fold_idx}.npy", scaler_dk_y)

        # load all variables from npy files
        dk_loaded = np.load(f"dk_{name}_{fold_idx}.npy")
        dl_loaded = np.load(f"dl_{name}_{fold_idx}.npy")
        y_train_k_loaded = np.load(f"y_train_k_{name}_{fold_idx}.npy")
        y_train_l_loaded = np.load(f"y_train_l_{name}_{fold_idx}.npy")
        x_test_loaded = np.load(f"x_test_{name}_{fold_idx}.npy")
        y_test_loaded = np.load(f"y_test_{name}_{fold_idx}.npy")
        x_train_loaded = np.load(f"x_train_{name}_{fold_idx}.npy")
        y_train_loaded = np.load(f"y_train_{name}_{fold_idx}.npy")
        x_train_transformed_loaded = np.load(f"x_train_transformed_{name}_{fold_idx}.npy")
        x_test_transformed_whole_loaded = np.load(f"x_test_transformed_whole_{name}_{fold_idx}.npy")
        y_train_transformed_loaded = np.load(f"y_train_transformed_{name}_{fold_idx}.npy")
        y_test_transformed_loaded = np.load(f"y_test_transformed_{name}_{fold_idx}.npy")
        dk_transformed_loaded = np.load(f"dk_transformed_{name}_{fold_idx}.npy")
        dl_transformed_loaded = np.load(f"dl_transformed_{name}_{fold_idx}.npy")
        x_test_transformed_dk_loaded = np.load(f"x_test_transformed_dk_{name}_{fold_idx}.npy")
        x_train_transformed_dk_loaded = np.load(f"x_train_transformed_dk_{name}_{fold_idx}.npy")
        y_train_k_transformed_loaded = np.load(f"y_train_k_transformed_{name}_{fold_idx}.npy")
        y_train_l_transformed_loaded = np.load(f"y_train_l_transformed_{name}_{fold_idx}.npy")
        y_test_transformed_dk_loaded = np.load(f"y_test_transformed_dk_{name}_{fold_idx}.npy")


        assert np.allclose(dk_loaded , dk) , f"dk_loaded {dk_loaded} , dk {dk}"
        assert np.allclose(dl_loaded , dl) , f"dl_loaded {dl_loaded} , dl {dl}"
        assert np.allclose(y_train_k_loaded , y_train_k) , f"y_train_k_loaded {y_train_k_loaded} , y_train_k {y_train_k}"
        assert np.allclose(y_train_l_loaded , y_train_l) , f"y_train_l_loaded {y_train_l_loaded} , y_train_l {y_train_l}"
        assert np.allclose(x_test_loaded , x_test) , f"x_test_loaded {x_test_loaded} , x_test {x_test}"
        assert np.allclose(y_test_loaded , y_test) , f"y_test_loaded {y_test_loaded} , y_test {y_test}"
        assert np.allclose(x_train_loaded , x_train) , f"x_train_loaded {x_train_loaded} , x_train {x_train}"
        assert np.allclose(y_train_loaded , y_train) , f"y_train_loaded {y_train_loaded} , y_train {y_train}"
        assert np.allclose(x_train_transformed_loaded , x_train_transformed) , f"x_train_transformed_loaded {x_train_transformed_loaded} , x_train_transformed {x_train_transformed}"
        assert np.allclose(x_test_transformed_whole_loaded , x_test_transformed_whole) , f"x_test_transformed_whole_loaded {x_test_transformed_whole_loaded} , x_test_transformed_whole {x_test_transformed_whole}"
        assert np.allclose(y_train_transformed_loaded , y_train_transformed) , f"y_train_transformed_loaded {y_train_transformed_loaded} , y_train_transformed {y_train_transformed}"
        assert np.allclose(y_test_transformed_loaded , y_test_transformed) , f"y_test_transformed_loaded {y_test_transformed_loaded} , y_test_transformed {y_test_transformed}"
        assert np.allclose(dk_transformed_loaded , dk_transformed) , f"dk_transformed_loaded {dk_transformed_loaded} , dk_transformed {dk_transformed}"
        assert np.allclose(dl_transformed_loaded , dl_transformed) , f"dl_transformed_loaded {dl_transformed_loaded} , dl_transformed {dl_transformed}"
        assert np.allclose(x_test_transformed_dk_loaded , x_test_transformed_dk) , f"x_test_transformed_dk_loaded {x_test_transformed_dk_loaded} , x_test_transformed_dk {x_test_transformed_dk}"
        assert np.allclose(x_train_transformed_dk_loaded , x_train_transformed_dk) , f"x_train_transformed_dk_loaded {x_train_transformed_dk_loaded} , x_train_transformed_dk {x_train_transformed_dk}"
        assert np.allclose(y_train_k_transformed_loaded , y_train_k_transformed) , f"y_train_k_transformed_loaded {y_train_k_transformed_loaded} , y_train_k_transformed {y_train_k_transformed}"
        assert np.allclose(y_train_l_transformed_loaded , y_train_l_transformed) , f"y_train_l_transformed_loaded {y_train_l_transformed_loaded} , y_train_l_transformed {y_train_l_transformed}"
        assert np.allclose(y_test_transformed_dk_loaded , y_test_transformed_dk) , f"y_test_transformed_dk_loaded {y_test_transformed_dk_loaded} , y_test_transformed_dk {y_test_transformed_dk}"
    





        dist_train_cobra_original = np.load(f"dist_train_cobra_original_{name}_{fold_idx}.npy")
        dist_test_cobra_original = np.load(f"dist_test_cobra_original_{name}_{fold_idx}.npy")
        dist_train_cobra_split = np.load(f"dist_train_cobra_split_{name}_{fold_idx}.npy")
        dist_test_cobra_split = np.load(f"dist_test_cobra_split_{name}_{fold_idx}.npy")
        dist_train_cobra_no_split = np.load(f"dist_train_cobra_no_split_{name}_{fold_idx}.npy")
        dist_test_cobra_no_split = np.load(f"dist_test_cobra_no_split_{name}_{fold_idx}.npy")



        # Original
        y_sir, y_sir_kernel , time_taken_in_search, time_taken_in_prediction = get_pred_sir_original(dist_train_cobra_original, dist_test_cobra_original, y_train_l.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'sir', 'mse': mean_squared_error(y_test, y_sir), 'R2': r2_score(y_test, y_sir) , 'data_trained_on': 'original', 'data_tested_on': 'original', 'time_taken_fit': time_taken_in_search, 'time_taken_infer': time_taken_in_prediction}, ignore_index=True)
        y_sir_split, y_sir_kernel_split , time_taken_in_search_split, time_taken_in_prediction_split = get_pred_sir_original(dist_train_cobra_split, dist_test_cobra_split, y_train.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'sir', 'mse': mean_squared_error(y_test, y_sir_split), 'R2': r2_score(y_test, y_sir_split) , 'data_trained_on': 'split', 'data_tested_on': 'split', 'time_taken_fit': time_taken_in_search_split, 'time_taken_infer': time_taken_in_prediction_split}, ignore_index=True)
        y_sir_no_split, y_sir_kernel_no_split , time_taken_in_search_no_split, time_taken_in_prediction_no_split = get_pred_sir_original(dist_train_cobra_no_split, dist_test_cobra_no_split, y_train.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'sir', 'mse': mean_squared_error(y_test, y_sir_no_split), 'R2': r2_score(y_test, y_sir_no_split) , 'data_trained_on': 'no_split', 'data_tested_on': 'no_split', 'time_taken_fit': time_taken_in_search_no_split, 'time_taken_infer': time_taken_in_prediction_no_split}, ignore_index=True)

        # Split Full prox
        y_cobra, y_cobra_kernel , time_taken_in_search_cobra, time_taken_in_prediction_cobra = get_pred_cobra_original(dist_train_cobra_original, dist_test_cobra_original, y_train_l.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'cobra', 'mse': mean_squared_error(y_test, y_cobra), 'R2': r2_score(y_test, y_cobra) , 'data_trained_on': 'original', 'data_tested_on': 'original', 'time_taken_fit': time_taken_in_search_cobra, 'time_taken_infer': time_taken_in_prediction_cobra}, ignore_index=True)
        y_cobra_split, y_cobra_kernel_split , time_taken_in_search_cobra_split, time_taken_in_prediction_cobra_split = get_pred_cobra_original(dist_train_cobra_split, dist_test_cobra_split, y_train.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'cobra', 'mse': mean_squared_error(y_test, y_cobra_split), 'R2': r2_score(y_test, y_cobra_split) , 'data_trained_on': 'split', 'data_tested_on': 'split', 'time_taken_fit': time_taken_in_search_cobra_split, 'time_taken_infer': time_taken_in_prediction_cobra_split}, ignore_index=True)
        y_cobra_no_split, y_cobra_kernel_no_split , time_taken_in_search_cobra_no_split, time_taken_in_prediction_cobra_no_split = get_pred_cobra_original(dist_train_cobra_no_split, dist_test_cobra_no_split, y_train.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'cobra', 'mse': mean_squared_error(y_test, y_cobra_no_split), 'R2': r2_score(y_test, y_cobra_no_split) , 'data_trained_on': 'no_split', 'data_tested_on': 'no_split', 'time_taken_fit': time_taken_in_search_cobra_no_split, 'time_taken_infer': time_taken_in_prediction_cobra_no_split}, ignore_index=True)

        # No Split
        y_gradient, y_gradient_kernel , time_taken_in_search_gradient, time_taken_in_prediction_gradient = get_pred_gradient_original(dist_train_cobra_original, dist_test_cobra_original, y_train_l.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'gradient', 'mse': mean_squared_error(y_test, y_gradient), 'R2': r2_score(y_test, y_gradient) , 'data_trained_on': 'original', 'data_tested_on': 'original', 'time_taken_fit': time_taken_in_search_gradient, 'time_taken_infer': time_taken_in_prediction_gradient}, ignore_index=True)
        y_gradient_split, y_gradient_kernel_split , time_taken_in_search_gradient_split, time_taken_in_prediction_gradient_split = get_pred_gradient_original(dist_train_cobra_split, dist_test_cobra_split, y_train.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'gradient', 'mse': mean_squared_error(y_test, y_gradient_split), 'R2': r2_score(y_test, y_gradient_split) , 'data_trained_on': 'split', 'data_tested_on': 'split', 'time_taken_fit': time_taken_in_search_gradient_split, 'time_taken_infer': time_taken_in_prediction_gradient_split}, ignore_index=True)
        y_gradient_no_split, y_gradient_kernel_no_split , time_taken_in_search_gradient_no_split, time_taken_in_prediction_gradient_no_split = get_pred_gradient_original(dist_train_cobra_no_split, dist_test_cobra_no_split, y_train.flatten())
        result_df.append({'dataset': name, 'fold': fold_idx, 'model': 'gradient', 'mse': mean_squared_error(y_test, y_gradient_no_split), 'R2': r2_score(y_test, y_gradient_no_split) , 'data_trained_on': 'no_split', 'data_tested_on': 'no_split', 'time_taken_fit': time_taken_in_search_gradient_no_split, 'time_taken_infer': time_taken_in_prediction_gradient_no_split}, ignore_index=True)




result_df.to_csv("cobra_result.csv")



                    

        
