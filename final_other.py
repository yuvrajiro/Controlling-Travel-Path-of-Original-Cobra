import gc
from uci_datasets import all_datasets
from custom_dataset import customDataset as Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import time
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

data_names = ['california']
models = [LassoCV(), RidgeCV(), DecisionTreeRegressor()]
n_grid = 100
n_iter = 100


# shut off future warnings
# import dataconversionwarning
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# DataConversionWarning
warnings.simplefilter(action='ignore', category=DataConversionWarning)

print(f"Starting Analysis for {data_names} datasets")
result_df = pd.DataFrame(columns=['dataset', 'fold', 'model', 'mse', 'R2' , 'data_trained_on', 'data_tested_on', 'time_taken_fit', 'time_taken_infer'])
data_index = 1
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



        # training the model on d_k
        dk_trained_model = []
        for model in models:
            fit_started = time.time()
            model.fit(dk_transformed, y_train_k_transformed)
            fit_ended = time.time()
            fitting_time = fit_ended - fit_started
            # test performance
            infer_started = time.time()
            y_pred = model.predict(dl_transformed)
            y_pred = scaler_dk_y.inverse_transform(y_pred.reshape(-1, 1))
            infer_ended = time.time()
            infer_time = infer_ended - infer_started
            mse = mean_squared_error(y_train_l, y_pred)
            r2 = r2_score(y_train_l, y_pred)
            result_df = result_df.append({'dataset': name, 'fold': fold_idx, 'model': model, 'mse': mse, 'R2': r2, 'data_trained_on': 'd_k', 'data_tested_on': 'd_l', 'time_taken_fit': fitting_time, 'time_taken_infer': infer_time}, ignore_index=True)
            infer_started = time.time()
            y_pred = model.predict(x_test_transformed_dk)
            infer_ended = time.time()
            infer_time = infer_ended - infer_started
            y_pred = scaler_dk_y.inverse_transform(y_pred.reshape(-1, 1))
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            result_df = result_df.append({'dataset': name, 'fold': fold_idx, 'model': model, 'mse': mse, 'R2': r2, 'data_trained_on': 'd_k', 'data_tested_on': 'test', 'time_taken_fit': fitting_time, 'time_taken_infer': infer_time}, ignore_index=True)
            # train performance
            infer_started = time.time()
            y_pred = model.predict(dk_transformed)
            infer_ended = time.time()
            infer_time = infer_ended - infer_started
            y_pred = scaler_dk_y.inverse_transform(y_pred.reshape(-1, 1))
            mse = mean_squared_error(y_train_k, y_pred)
            r2 = r2_score(y_train_k, y_pred)
            result_df = result_df.append({'dataset': name, 'fold': fold_idx, 'model': model, 'mse': mse, 'R2': r2, 'data_trained_on': 'd_k', 'data_tested_on': 'd_k', 'time_taken_fit': fitting_time, 'time_taken_infer': infer_time}, ignore_index=True)
            dk_trained_model.append(model)
            # performance on x_train_transformed_dk
            infer_started = time.time()
            y_pred = model.predict(x_train_transformed_dk)
            infer_ended = time.time()
            infer_time = infer_ended - infer_started
            y_pred = scaler_dk_y.inverse_transform(y_pred.reshape(-1, 1))
            mse = mean_squared_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)
            result_df = result_df.append({'dataset': name, 'fold': fold_idx, 'model': model, 'mse': mse, 'R2': r2, 'data_trained_on': 'd_k', 'data_tested_on': 'train', 'time_taken_fit': fitting_time, 'time_taken_infer': infer_time}, ignore_index=True)
        # train the model on whole dataset
        full_trained_model = []
        for model in models:
            fit_started = time.time()
            model.fit(x_train_transformed, y_train_transformed)
            fit_ended = time.time()
            fitting_time = fit_ended - fit_started
            # test performance
            infer_started = time.time()
            y_pred = model.predict(x_test_transformed_whole)
            
            infer_ended = time.time()
            y_pred = scaler_whole_y.inverse_transform(y_pred.reshape(-1, 1))
            infer_time = infer_ended - infer_started
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            result_df = result_df.append({'dataset': name, 'fold': fold_idx, 'model': model, 'mse': mse, 'R2': r2, 'data_trained_on': 'train', 'data_tested_on': 'xtest', 'time_taken_fit': fitting_time, 'time_taken_infer': infer_time}, ignore_index=True)
            # train performance
            infer_started = time.time()
            y_pred = model.predict(x_train_transformed)
            infer_ended = time.time()
            infer_time = infer_ended - infer_started
            y_pred = scaler_whole_y.inverse_transform(y_pred.reshape(-1, 1))
            mse = mean_squared_error(y_train, y_pred)
            r2 = r2_score(y_train, y_pred)
            result_df = result_df.append({'dataset': name, 'fold': fold_idx, 'model': model, 'mse': mse, 'R2': r2, 'data_trained_on': 'train', 'data_tested_on': 'test', 'time_taken_fit': fitting_time, 'time_taken_infer': infer_time}, ignore_index=True)
            full_trained_model.append(model)

        np.save(f"dk_{name}_{fold_idx}.npy", dk)
        np.save(f"dl_{name}_{fold_idx}.npy", dl)
        np.save(f"y_train_k_{name}_{fold_idx}.npy", y_train_k)
        np.save(f"y_train_l_{name}_{fold_idx}.npy", y_train_l)
        np.save(f"x_test_{name}_{fold_idx}.npy", x_test)
        np.save(f"y_test_{name}_{fold_idx}.npy", y_test)
        np.save(f"x_train_{name}_{fold_idx}.npy", x_train)
        np.save(f"y_train_{name}_{fold_idx}.npy", y_train)
        np.save(f"x_train_transformed_{name}_{fold_idx}.npy", x_train_transformed)
        np.save(f"x_test_transformed_whole_{name}_{fold_idx}.npy", x_test_transformed_whole)
        np.save(f"y_train_transformed_{name}_{fold_idx}.npy", y_train_transformed)
        np.save(f"y_test_transformed_{name}_{fold_idx}.npy", y_test_transformed)
        np.save(f"dk_transformed_{name}_{fold_idx}.npy", dk_transformed)
        np.save(f"dl_transformed_{name}_{fold_idx}.npy", dl_transformed)
        np.save(f"x_test_transformed_dk_{name}_{fold_idx}.npy", x_test_transformed_dk)
        np.save(f"x_train_transformed_dk_{name}_{fold_idx}.npy", x_train_transformed_dk)
        np.save(f"y_train_k_transformed_{name}_{fold_idx}.npy", y_train_k_transformed)
        np.save(f"y_train_l_transformed_{name}_{fold_idx}.npy", y_train_l_transformed)
        np.save(f"y_test_transformed_dk_{name}_{fold_idx}.npy", y_test_transformed_dk)
        np.save(f"scaler_whole_{name}_{fold_idx}.npy", scaler_whole)
        np.save(f"scaler_whole_y_{name}_{fold_idx}.npy", scaler_whole_y)
        np.save(f"scaler_dk_{name}_{fold_idx}.npy", scaler_dk)
        np.save(f"scaler_dk_y_{name}_{fold_idx}.npy", scaler_dk_y)


        # prediction on d_l by all macchines trained on d_k (Original COBRA)
        pred_dl_train_dk_ = np.zeros((len(dl), len(models)))
        pred_test_train_dk_ = np.zeros((len(x_test_transformed_dk), len(models)))

        # split full prox
        pred_train_train_dk_ = np.zeros((len(x_train_transformed_dk), len(models)))
        #pred_test_train_dk_ = np.zeros((len(x_test_transformed_dk), len(models)))

        # no split
        pred_train_train_train = np.zeros((len(x_train_transformed), len(models)))
        pred_test_train_train = np.zeros((len(x_test_transformed_whole), len(models)))

        for i, model in enumerate(dk_trained_model):
            pred_dl_train_dk_[:, i] = model.predict(dl_transformed).ravel()
            pred_test_train_dk_[:, i] = model.predict(x_test_transformed_dk).ravel()
            pred_train_train_dk_[:, i] = model.predict(x_train_transformed_dk).ravel()
            

        for i, model in enumerate(full_trained_model):
            pred_train_train_train[:, i] = model.predict(x_train_transformed).ravel()
            pred_test_train_train[:, i] = model.predict(x_test_transformed_whole).ravel()

        dist_train_cobra_original = np.zeros((len(pred_dl_train_dk_), len(pred_dl_train_dk_) , len(models)))
        dist_test_cobra_original = np.zeros((len(pred_test_train_dk_), len(pred_dl_train_dk_) , len(models)))

        dist_train_cobra_split = np.zeros((len(pred_train_train_dk_), len(pred_train_train_dk_) , len(models)))
        dist_test_cobra_split = np.zeros((len(pred_test_train_train), len(pred_train_train_dk_) , len(models)))

        dist_train_cobra_no_split = np.zeros((len(pred_train_train_train), len(pred_train_train_train) , len(models)))
        dist_test_cobra_no_split = np.zeros((len(pred_test_train_train), len(pred_train_train_train) , len(models)))

        # pred_dl_train_dk_ is n_samples x n_models shape array  calculate l1 distance between all pairs for all models
        dist_train_cobra_original = np.abs(pred_dl_train_dk_[:, None, :] - pred_dl_train_dk_[None, :, :]) #y_train_l
        np.save(f"dist_train_cobra_original_{name}_{fold_idx}.npy", dist_train_cobra_original)
        del dist_train_cobra_original
        gc.collect()
        dist_test_cobra_original = np.abs(pred_test_train_dk_[:, None, :] - pred_dl_train_dk_[None, :, :]) #y_test
        np.save(f"dist_test_cobra_original_{name}_{fold_idx}.npy", dist_test_cobra_original)
        del dist_test_cobra_original
        gc.collect()

        dist_train_cobra_split = np.abs(pred_train_train_dk_[:, None, :] - pred_train_train_dk_[None, :, :]) #y_train
        np.save(f"dist_train_cobra_split_{name}_{fold_idx}.npy", dist_train_cobra_split)
        del dist_train_cobra_split
        gc.collect()
        dist_test_cobra_split = np.abs(pred_test_train_train[:, None, :] - pred_train_train_dk_[None, :, :])
        np.save(f"dist_test_cobra_split_{name}_{fold_idx}.npy", dist_test_cobra_split)
        del dist_test_cobra_split
        gc.collect()

        dist_train_cobra_no_split = np.abs(pred_train_train_train[:, None, :] - pred_train_train_train[None, :, :])
        np.save(f"dist_train_cobra_no_split_{name}_{fold_idx}.npy", dist_train_cobra_no_split)
        del dist_train_cobra_no_split
        gc.collect()
        dist_test_cobra_no_split = np.abs(pred_test_train_train[:, None, :] - pred_train_train_train[None, :, :])
        np.save(f"dist_test_cobra_no_split_{name}_{fold_idx}.npy", dist_test_cobra_no_split)
        del dist_test_cobra_no_split
        gc.collect() 



result_df.to_csv('results_baseline_california.csv')

                    

        
