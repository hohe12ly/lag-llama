# zero-shot lag-llama inference on PSML minute data

# %%
# YL
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys # YL
sys.path.insert(0, '/mnt/data/home/yxl/test/lagllama/lag-llama') # YL
import argparse

from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm.autonotebook import tqdm

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator

# YL
from gluonts.dataset.common import ListDataset, load_datasets
from gluonts.evaluation._base import aggregate_valid

from lightning.pytorch.loggers import CSVLogger

import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np
import random
import pickle

#dataset_path = Path("/home/toolkit/datasets")
dataset_path = Path("/mnt/data/home/yxl/test/test_ai/datasets")

# %%
def print_hyperparameters(ckpt):
    print("=== Hyperparameters ===")
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    print(estimator_args)

# zeroshot function
# old format: def get_lag_llama_predictions(f_lagllama_pretrained, dataset, prediction_length, num_samples=100):
def zeroshot(name, logdir, f_lagllama_pretrained, dataset, prediction_length, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load(f_lagllama_pretrained, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    logdir = logdir + '/' + name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    logger = CSVLogger(
        save_dir=logdir #,
        #flush_logs_every_n_steps=1,
        #version=lightning_version_to_use
    )

    estimator = LagLlamaEstimator(
        ckpt_path=f_lagllama_pretrained,
        prediction_length=prediction_length,
        context_length=context_length,
        #lags_seq=["Q", "M", "W", "D", "H"], # hourly data doesn't need T, S. TODO: ckpt has diff dims, causing tensor dim error

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],

        nonnegative_pred_samples=nonnegative_pred_samples,

        # linear positional encoding scaling
        rope_scaling={
            "type": "linear", # "dynamic", "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },

        batch_size=batch_size,
        num_parallel_samples=num_samples,
        trainer_kwargs = { "logger": logger }, # <- lightning trainer arguments
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss

# fewshots function
def fewshots(name, logdir, f_lagllama_pretrained, dataset_train, dataset_test, prediction_length, dataset_val=None, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True, max_epochs=50):
    ckpt = torch.load(f_lagllama_pretrained, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    print_hyperparameters(ckpt)

    logdir = logdir + '/' + name
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logger = CSVLogger(
        save_dir=logdir #,
        #flush_logs_every_n_steps=1,
        #version=lightning_version_to_use
    )

    estimator = LagLlamaEstimator(
            ckpt_path=f_lagllama_pretrained,
            prediction_length=prediction_length,
            context_length=context_length,
            #lags_seq=["Q", "M", "W", "D", "H"], # hourly data doesn't need T, S. TODO: ckpt has diff dims, causing tensor dim error

            # distr_output="neg_bin",
            # scaling="mean",
            nonnegative_pred_samples=nonnegative_pred_samples,
            aug_prob=0, #0.5, # YL
            lr=5e-4,

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            time_feat=estimator_args["time_feat"],

            rope_scaling={
                "type": "linear", #"dynamic", 
                "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            },

            batch_size=batch_size, # YL
            num_parallel_samples=num_samples,
            trainer_kwargs = {
                "max_epochs": max_epochs,
                "logger": logger,
            }, # <- lightning trainer arguments
        )    
    if dataset_val is not None:
        predictor = estimator.train(
            training_data=dataset_train,
            validation_data=dataset_val, 
            cache_data=True, 
            shuffle_buffer_length=1000
        )
    else:
        predictor = estimator.train(training_data=dataset_train, cache_data=True, shuffle_buffer_length=1000)    
    
    modeldir = logdir + '/' + "model-fewshots-"+name
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    predictor.serialize(Path(modeldir)) # save to disk, load using gluonts.torch.model.predictor.PyTorchPredictor.deserialize(Path, device)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset_test,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset_test), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset_test), desc="Ground truth"))

    return forecasts, tss

# %%
# YL start
powergrid_dataset_names = ['IEEE_33bus_DER_10year_hourly']
#powergrid_ts_targets = ['voltage'] # test why few shots is not working on voltage
powergrid_ts_targets = ['load_profile', 'load_dispatch'] #, 'voltage'] #, 'pv_output', 'dg_output']
powergrid_dataset_names = [dname + '_' + ts_target for dname in powergrid_dataset_names for ts_target in powergrid_ts_targets]
powergrid_num_buses = 33
powergrid_graph_nodes = ['bus_' + str(i+1) for i in range(powergrid_num_buses)] # index 0 is bus_1
rolling_evaluations = 5 # num of rolling evaluations in the test dataset
powergrid_graph_nodes_by_category = {
    'PV': [11, 27], # bus_12, bus_28
    'MT': [17, 32], # bus_18, bus_33
    'SVC': [9, 15, 29], # bus_10, bus_16, bus_30
    'EV': [7, 12, 14, 28], # bus_8, bus_13, bus_15, bus_29
    'HVAC': [8, 13, 16, 20, 30], # bus_9, bus_14, bus_17, bus_21, bus_31
}

def get_powergrid_graph_nodes_category(powergrid_num_buses, powergrid_graph_nodes_by_category):
    category = [ 'Idle' for i in range(powergrid_num_buses) ]
    for cat, nodes in powergrid_graph_nodes_by_category.items():
        for node in nodes:
            category[node] = cat
    return category
powergrid_graph_nodes_cat = get_powergrid_graph_nodes_category(powergrid_num_buses, powergrid_graph_nodes_by_category)

# buses that are trained and tested together. multiple ways to combine buses
# powergrid_tsset = { # same as bus category
#     'HVAC': [13, 16, 20, 30], # bus_14, bus_17, bus_21, bus_31. bus_9 has no load
#     'EV': [12, 14, 28], # bus_13, bus_15, bus_29. bus_8 has no load
#     'PV': [27], # bus_28. bus_12 has no load
#     'MT': [32], # bus_33. bus_18 has no load
#     'SVC': [15, 29], # bus_16, bus_30. bus_10 has no load
# }
#powergrid_tsset = { powergrid_graph_nodes[i]: [i] for i in [13, 16, 12, 27, 32, 15] }
#powergrid_tsset = { powergrid_graph_nodes[i]: [i] for i in [13, 16, 20, 30] } # all HVAC buses with load
#powergrid_tsset = { 'entire_network': list(range(len(powergrid_graph_nodes))) } # all buses trained together
powergrid_tsset = { powergrid_graph_nodes[i]: [i] for i in [12, 16, 32] } # bus samples, bus by bus
powergrid_tsset_id = { k:i for i, k in enumerate(powergrid_tsset.keys()) }

def data_has_negative_values(f_data):
    d = load_datasets(
        metadata=f_data,
        train=f_data / "train",
        test=f_data / "test",
    )
    for dset in [d.train, d.test]:
        for x in dset:
            if np.any(x['target'] < 0):
                return True
    return False

def create_powergrid_sliding_window_dataset(name, window_size, node_indexes, ts_cat_id, num_windows=None, num_rolling_evals=1, is_train=True):
    data_type = 'train'
    if not is_train:
        data_type = 'val'
    psml_dataset_path = dataset_path / name
    d = load_datasets(
        metadata=psml_dataset_path,
        train=psml_dataset_path / "train",
        test=psml_dataset_path / "test",
    )
    global_id = 0
    data = ListDataset([], freq=d.metadata.freq)
    dataset = d.train if is_train else d.test
    for i_x, x in enumerate(dataset):
        rolling_index = i_x % num_rolling_evals
        node_index = (i_x - rolling_index) // num_rolling_evals # i_x = node_index * num_rolling_evals + rolling_index
        if node_index not in node_indexes:
            continue
        windows = []
        count_windows = 0
        for i in range(0, len(x['target']), window_size):
            windows.append({
                'target': x['target'][i:i+window_size],
                'start': x['start'] + i,
                'item_id': str(node_index), # each item_id is a different time series
                'feat_static_cat': np.array([ts_cat_id]),
            })
            global_id += 1
            count_windows += 1
            if num_windows is not None and count_windows >= num_windows:
                break
        data += ListDataset(windows, freq=d.metadata.freq)
        print(data_type, 'datagen: node start index', global_id - 1, 'w/', len(windows), 'windows, from', len(x['target']), 'timesteps')

    print(data_type, 'datagen: total', len(data), ' series w/ window_size ', window_size, ' each')
    return data

# only fetch one rolling evaluation
def create_powergrid_test_dataset(name, window_size, node_indexes, ts_cat_id, num_rolling_evals, rolling_eval_index=0):
    # Similar to `create_sliding_window_dataset` but for test dataset
    psml_dataset_path = dataset_path / name
    dataset = load_datasets(
        metadata=psml_dataset_path,
        train=psml_dataset_path / "train",
        test=psml_dataset_path / "test",
    )
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length

    data = []
    count = 0
    ts_len = 0
    for i_x, x in enumerate(dataset.test):
        rolling_index = i_x % num_rolling_evals
        node_index = (i_x - rolling_index) // num_rolling_evals # i_x = node_index * num_rolling_evals + rolling_index
        if rolling_index != rolling_eval_index:
            continue
        if node_index not in node_indexes:
            continue
        if ts_len == 0:
            ts_len = len(x['target'])
        assert ts_len == len(x['target'])
        offset = len(x['target']) - window_size - prediction_length
        if offset > 0:
            target = x['target'][-(window_size + prediction_length):]
            data.append({
                'target': target,
                'start': x['start'] + offset,
                'item_id': str(node_index), # each item_id is a different time series
                'feat_static_cat': np.array([ts_cat_id]),
            })
        else:
            data.append(x)
        count += 1
        if count >= len(node_indexes):
            break
    print('test datagen: ', len(data), ' series w/ window_size ', window_size, ' each')
    return ListDataset(data, freq=freq), prediction_length

def plot_pred_results(forecasts, tss, name, ts_cat, ts_set, odir, nodelist, prediction_length):
    '''
    plot forecast and ground truth as well quantile info on all graph nodes at the `rolling_eval_index`th 
    rolling evaluation with `prediction_width`. `forecasts` and `tss` is a linear list of time series data.
    the index i is determined in the order of `[node_index, rolling_eval_index]` or 
    `[rolling_eval_index, node_index]`, depending on `nodedim_is_first`. 

    unfortunately, info about `num_rolling_evals` and `num_nodes` is not stored in the dataset metadata. 
    so this function is dataset specific. you have to know the dataset structure to use this function.
    '''
    # psml test data order: (node/bus, rolling eval): timesteps data
    #num_buses = 23
    #num_rolling_evals = 6
    #prediction_length, num_buses, num_rolling_evals

    fig, axes = plt.subplots(len(ts_set), 1, figsize=(10, len(ts_set) * 5))
    #print('name', name, 'ts_cat', ts_cat, 'ts_set: ', ts_set)
    if (len(ts_set) == 1):
        axes = [axes]
    #data_index = [ (i * num_rolling_evals + rolling_eval_index) if nodedim_is_first else (i + rolling_eval_index * num_nodes) for i in range(num_nodes) ]
    # assume only one rolling evaluation
    data_index = list(range(len(ts_set)))
    #print('shape of each tss: ', [ (i, data_index[i], tss[data_index[i]].shape) for i in range(num_nodes) ] )
    
    for i, ax in enumerate(axes): # i is ts_set index
        plot_data = tss[data_index[i]][-(prediction_length * 2):]
        data_min, data_max = plot_data.values.reshape(-1).min(), plot_data.values.reshape(-1).max()
        #print('test data min, max: ', data_min, data_max)
        ax.set_ylim(data_min * 0.7, data_max * 1.3) # w/ 30% margin
        ax.plot(tss[data_index[i]][-(prediction_length * 2):].to_timestamp())
        plt.sca(ax)
        forecasts[data_index[i]].plot(intervals=(0.5, 0.8, 0.9, 0.95), color='m')
        plt.legend(['ground truth', 'pred mean', '0.5', '0.8', '0.9', '0.95'])
        plt.title('node: ' + str(ts_set[i]) + ' , name: ' + nodelist[ts_set[i]] + ', cat: ' + ts_cat)
    plt.savefig(odir + '/' + 'perf_pred_' + name + '_' +
                'cat_' + ts_cat + 
                #'_predlen_' + str(prediction_length) + 
                '.png')

    fig, axes = plt.subplots(len(ts_set), 1, figsize=(10, len(ts_set) * 5))
    if (len(ts_set) == 1):
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(tss[data_index[i]].to_timestamp())
        plt.sca(ax)
        plt.title('node: ' + str(ts_set[i]) + ' , name: ' + nodelist[ts_set[i]] + ', cat: ' + ts_cat + ' [timesteps: ' + str(len(tss[data_index[i]])) + ']')
    plt.savefig(odir + '/' + 'testdata_allbuses_' + name + '_' +
                'cat_' + ts_cat + 
                #'_predlen_' + str(prediction_length) + 
                '.png')
    
    # save data: using pickle
    with open(
        odir + '/' + 'forecast_' + name + '_' +
        'cat_' + ts_cat + 
        #'_predlen_' + str(prediction_length) + 
        '.pkl', 'wb') as f:
        pickle.dump(forecasts, f, pickle.HIGHEST_PROTOCOL)
    with open(
        odir + '/' + 'testdata_' + name + '_' +
        'cat_' + ts_cat + 
        #'_predlen_' + str(prediction_length) + 
        '.pkl', 'wb') as f:
        pickle.dump(tss, f, pickle.HIGHEST_PROTOCOL)
    # np.save(
    #     odir + '/' + 'testdata_allbuses_' + name + '_' +
    #     'cat_' + ts_cat + 
    #     '_predlen_' + str(prediction_length) + 
    #     '.npy'
    #     ,
    #    np.array([ tss[data_index[i]].values.reshape(-1) for i in range(len(ts_set)) ]) # shape of tss[i].values: (context_length + prediction_length,)
    # )
    # np.save(
    #     odir + '/' + 'forecastdata_allbuses_' + name + '_' +
    #     'cat_' + ts_cat + 
    #     '_predlen_' + str(prediction_length) + 
    #     '.npy'
    #     ,
    #     np.array([ forecasts[data_index[i]].samples for i in range(len(ts_set)) ]) # shape of forecasts[i].samples: (num_samples, prediction_length)
    # )


# YL end

# %%
def inference(f_lagllama_pretrained, log_dir, is_fewshots=True):
    if is_fewshots:
        log_dir = log_dir + '_fewshots'
    else:
        log_dir = log_dir + '_zeroshots'
    print("log_dir : ", log_dir)
    print("os.path.exists(log_dir) : ", os.path.exists(log_dir))

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    num_samples = 20
    #for name in powergrid_dataset_names[:1]: # DEBUG: only run on the first dataset
    for name in powergrid_dataset_names:
        for ts_cat, ts_set in powergrid_tsset.items():
            print(f'Predict on {name}')
            test_data, prediction_length = create_powergrid_test_dataset(name, 24 * 365, # look back 1 year
                                                                         node_indexes=ts_set, ts_cat_id=powergrid_tsset_id[ts_cat], 
                                                                         num_rolling_evals=rolling_evaluations, 
                                                                         rolling_eval_index=0) # default is 0. use 3 for voltage. rolling eval 0 has weird test data on voltage

            print(f'{name} prediction length: {prediction_length}')

            if not is_fewshots:
                forecasts, tss = zeroshot(name, log_dir, f_lagllama_pretrained, test_data, prediction_length, num_samples=num_samples, context_length=24)
            else: 
                train_data = create_powergrid_sliding_window_dataset(name, 24 * 365, node_indexes=ts_set, ts_cat_id=powergrid_tsset_id[ts_cat], num_windows=None, 
                                                                     num_rolling_evals=1, is_train=True) # num_rolling_evals must be 1 for training data
                val_data = create_powergrid_sliding_window_dataset(name, 24 * 365, node_indexes=ts_set, ts_cat_id=powergrid_tsset_id[ts_cat], 
                                                                   num_windows=None, num_rolling_evals=rolling_evaluations, is_train=False)
                forecasts, tss = fewshots(name, log_dir, f_lagllama_pretrained, train_data, test_data, prediction_length, dataset_val=val_data, num_samples=num_samples, context_length=24, max_epochs=50)

            plot_pred_results(forecasts, tss, name, ts_cat, ts_set, log_dir, powergrid_graph_nodes, prediction_length)

            #evaluator = Evaluator(num_workers=1, aggregation_strategy=aggregate_valid)
            evaluator = Evaluator()
            agg_metrics, item_metrics = evaluator(
                iter(tss), iter(forecasts), num_series=len(test_data)
            )

            with open(f'{log_dir}/{name}_{ts_cat}.json', 'w') as f:
                json.dump(agg_metrics, f)

            item_metrics.to_csv(f'{log_dir}/{name}_{ts_cat}_item_metrics.csv')

# %%
# command examples
# python fewshots_IEEE_33bus_DER_10year_hourly.py --jobdir /mnt/data/home/yxl/test/lagllama/lag-llama/runs --jobname IEEE_33bus_DER_10year_hourly_test_1 --pretrained-model /mnt/data/home/yxl/test/lagllama/lag-llama/pretrained/lag-llama.ckpt #--is-fewshots
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobdir", type=str, required=True)
    parser.add_argument("--jobname", type=str, required=True)
    parser.add_argument("--pretrained-model", type=str, required=True)
    # parser.add_argument("--context_length", type=int, default=256)
    # parser.add_argument("--n_layer", type=int, default=4)
    # parser.add_argument("--n_embd", type=int, default=256)
    # parser.add_argument("--n_head", type=int, default=4)
    # parser.add_argument("--aug_prob", type=float, default=0.5)
    # parser.add_argument("--aug_rate", type=float, default=0.1)
    # parser.add_argument("--batch_size", type=int, default=100)
    # parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    # # estimator trainer kwarg args
    # parser.add_argument("--limit_val_batches", type=int, default=10)
    # parser.add_argument("--max_epochs", type=int, default=1000)
    #parser.add_argument("--gpu", type=int, default=0)
    # Other args
    parser.add_argument('--is-zeroshot', action='store_true')
    # Model
    args = parser.parse_args()
    print("YL arguments:")
    print(args)
    inference(
        args.pretrained_model, #'/mnt/data/home/yxl/test/lagllama/lag-llama/pretrained/lag-llama.ckpt',
        args.jobdir + '/' + args.jobname, #/mnt/data/home/yxl/test/lagllama/lag-llama/runs/IEEE_33bus_DER_10year_hourly_test_1',
        is_fewshots=not args.is_zeroshot
    )

# %%
# def get_lag_llama_predictions(f_lagllama_pretrained, dataset, prediction_length, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
#     ckpt = torch.load(f_lagllama_pretrained, map_location=device)
#     estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

#     estimator = LagLlamaEstimator(
#         ckpt_path=f_lagllama_pretrained,
#         prediction_length=prediction_length,
#         context_length=context_length,

#         # estimator args
#         input_size=estimator_args["input_size"],
#         n_layer=estimator_args["n_layer"],
#         n_embd_per_head=estimator_args["n_embd_per_head"],
#         n_head=estimator_args["n_head"],
#         scaling=estimator_args["scaling"],
#         time_feat=estimator_args["time_feat"],

#         nonnegative_pred_samples=nonnegative_pred_samples,

#         # linear positional encoding scaling
#         rope_scaling={
#             "type": "linear",
#             "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
#         },

#         batch_size=batch_size,
#         num_parallel_samples=num_samples,
#     )

#     lightning_module = estimator.create_lightning_module()
#     transformation = estimator.create_transformation()
#     predictor = estimator.create_predictor(transformation, lightning_module)

#     forecast_it, ts_it = make_evaluation_predictions(
#         dataset=dataset,
#         predictor=predictor,
#         num_samples=num_samples
#     )
#     forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
#     tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

#     return forecasts, tss
# %%
# name = psml_dataset_names[0]
# test_data, prediction_length = create_psml_test_dataset(name, 256)# %%

# f_lagllama_pretrained = '/mnt/data/home/yxl/test/lagllama/lag-llama/pretrained/lag-llama.ckpt'
# log_dir = '/mnt/data/home/yxl/test/lagllama/lag-llama/runs/zeroshot_psml_minute'
# d_m4 = get_dataset("m4_weekly")
# f_lagllama_pretrained = '/mnt/data/home/yxl/test/lagllama/lag-llama/pretrained/lag-llama.ckpt'
# log_dir = '/mnt/data/home/yxl/test/lagllama/lag-llama/runs/zeroshot_mr4_weekly'
# forecasts, tss = get_lag_llama_predictions(f_lagllama_pretrained, d_m4.test, d_m4.metadata.prediction_length, num_samples=20)
# # %%
# type(forecasts), type(tss), type(forecasts[0]), type(tss[0]) 
# # output: list, list, gluonts.model.forecast.SampleForecast, pandas.core.frame.DataFrame
# # %%
# type(forecasts[0].samples), type(tss[0].values)
# # %%
# type(forecasts[0].mean), forecasts[0].mean.shape
# # %%
# forecasts[0].samples.shape
# # %%
# tss[0].shape
# # %%
# forecasts[0].mean
# # %%
# tss[0].values.reshape(-1).shape
# import pickle
# tmpdir='/mnt/data/home/yxl/tmp'
# with open(tmpdir + '/forecasts.pkl', 'wb') as f:
#     pickle.dump(forecasts, f, pickle.HIGHEST_PROTOCOL)
# # %%
# with open(tmpdir + '/tss.pkl', 'wb') as f:
#     pickle.dump(tss, f, pickle.HIGHEST_PROTOCOL)
# # %%
# with open(tmpdir + '/forecasts.pkl', 'rb') as f:
#     forecasts2 = pickle.load(f)
# with open(tmpdir + '/tss.pkl', 'rb') as f:
#     tss2 = pickle.load(f)
# # %%
# type(forecasts2), type(tss2), type(forecasts2[0]), type(tss2[0])
# # %%
# d_m4 = get_dataset("m4_weekly")
# # %%
# len(d_m4.train), len(d_m4.test) # (359, 359)
# # %%
# d_traffic = get_dataset("traffic")
# len(d_traffic.train), len(d_traffic.test) # (862, 6034). rolling evaluation periods = 7
# %%
## first run: the collab case from:
# https://colab.research.google.com/drive/1uvTmh-pe1zO5TeaaRVDdoEWJ5dFDI-pA?usp=sharing
# dataset = get_dataset("m4_weekly")
# prediction_length = dataset.metadata.prediction_length
# context_length = prediction_length*3
# num_samples = 20
# device = "cuda"
# %%
# check if the dataset has negative values
# for name in powergrid_dataset_names:
#     print(name, data_has_negative_values(dataset_path / name))
# IEEE_33bus_DER_10year_hourly_voltage False
# IEEE_33bus_DER_10year_hourly_load_profile False
# IEEE_33bus_DER_10year_hourly_load_dispatch False
# IEEE_33bus_DER_10year_hourly_pv_output False
# IEEE_33bus_DER_10year_hourly_dg_output False
# %%
