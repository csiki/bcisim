import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch
from torch.nn import Module, LSTM, Sequential, Linear, MSELoss, Conv2d, Flatten
from collections import OrderedDict

from neur_dec import NeurDec
from experiments.monkey_grasp.monkey_gen import *
from experiments.monkey_grasp.simple_lstm import *


if __name__ == '__main__':
    # convert monkey neural recordings into graphs
    # create a generator that outputs those graphs w/ future states and outcomes to be predicted
    monkey_name = 'Lilou'  # Lilou or Nikos2
    with open(f'{monkey_name}.pckl', 'rb') as f:
        monkey = pickle.load(f, encoding='bytes')

    nine_neighbor = np.ones((3, 3), dtype=int)
    four_neighbor = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=int)

    fs = 1000
    gen = MonkeyDataGen(monkey, four_neighbor, fs)

    # build training and test sets
    train_ratio = .7
    fields = ['spikes_t', 'forces_t']

    trial_indices = np.random.permutation(gen.ntrials)
    train_trials = trial_indices[:int(gen.ntrials * train_ratio)]
    test_trials = trial_indices[len(train_trials):]

    # data gen
    # train_spikes, train_forces = [], []
    # test_spikes, test_forces = [], []
    #
    # for trial_i in trial_indices:
    #     for i, s in enumerate(gen.vec_trial_gen(trial_i, fields)):
    #         if trial_i in train_trials:  # linear search but whatever, see below data gen for RNNs
    #             train_spikes.append(s[0])
    #             train_forces.append(s[1])
    #         else:
    #             test_spikes.append(s[0])
    #             test_forces.append(s[1])
    # train_spikes, train_forces = np.array(train_spikes), np.array(train_forces)
    # test_spikes, test_forces = np.array(test_spikes), np.array(test_forces)
    # print('train', train_spikes.shape, train_forces.shape)
    # print('test', test_spikes.shape, test_forces.shape)

    # # train linear regression - shit
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression(n_jobs=4).fit(train_spikes, train_forces.ravel())
    # train_score = model.score(train_spikes, train_forces.ravel())
    # test_score = model.score(test_spikes, test_forces.ravel())
    # print('linear train score', train_score, 'test score', test_score)

    # # lasso - shit
    # from sklearn.linear_model import LassoLars
    # model = LassoLars().fit(train_spikes, train_forces.ravel())
    # train_score = model.score(train_spikes, train_forces.ravel())
    # test_score = model.score(test_spikes, test_forces.ravel())
    # print('lasso train score', train_score, 'test score', test_score)

    # # simple NN - shit and slow
    # from sklearn.neural_network import MLPRegressor
    # model = MLPRegressor(hidden_layer_sizes=[64], activation='tanh', early_stopping=True, validation_fraction=.2) \
    #     .fit(train_spikes, train_forces.ravel())
    # train_score = model.score(train_spikes, train_forces.ravel())
    # test_score = model.score(test_spikes, test_forces.ravel())
    # print('simple NN train score', train_score, 'test score', test_score)

    # TODO Kalman filter: neural state is observed (y), and the force is hidden (x)
    # from pykalman import KalmanFilter, UnscentedKalmanFilter
    # model = KalmanFilter(n_dim_state=train_forces.shape[-1], n_dim_obs=train_spikes.shape[-1])
    # model.em(train_spikes, train_forces, n_iter=10)

    # RNN
    k = 64
    nepoch = 150
    dev = 'cuda'  # cuda | cpu
    grid_conv = True
    grid_unit_aggr_fun = lambda x: np.clip(x, 0, 1)

    # conv embedding: 10x10 -> 1xN
    grid_conv_chans = [8, 8, 8, 16]
    grid_embedding = GridConv(grid_conv_chans)

    hidden_dim = 16
    num_layers = 1
    model = SimpleLSTM(grid_conv_chans[-1] if grid_conv else gen.nunits, hidden_dim=hidden_dim, out_dim=1,
                       num_layers=num_layers, embedding=grid_embedding if grid_conv else None).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    loss_fun = MSELoss()
    model_name = f'rnn_model_grid-{int(grid_conv)}_{"-".join([str(g) for g in grid_conv_chans]) if grid_conv else ""}' \
                 f'_{nepoch}_{k}_{num_layers}_{hidden_dim}'

    print(f'epochs: {nepoch}, grid: {grid_conv}, grid chans: {grid_conv_chans}, k: {k}, dev: {dev}, '
          f'hidden: {hidden_dim}, num layers: {num_layers}')

    train_losses = []
    test_trial_losses = []
    for epoch in range(nepoch):
        if grid_conv:
            train_trial_gens = [gen.gen_grid_trial(trial_i, fields, grid_unit_aggr_fun) for trial_i in train_trials]
            test_trial_gens = [gen.gen_grid_trial(trial_i, fields, grid_unit_aggr_fun) for trial_i in test_trials]
        else:
            train_trial_gens = [gen.gen_vec_trial(trial_i, fields) for trial_i in train_trials]
            test_trial_gens = [gen.gen_vec_trial(trial_i, fields) for trial_i in test_trials]

        train_loss = model.tbptt_train(train_trial_gens, loss_fun, optimizer, k=64)
        train_losses.append(train_loss)
        test_losses = model.test(test_trial_gens, loss_fun)
        test_trial_losses.append(test_losses)
        print(f'{epoch}/{nepoch} losses', np.mean(test_losses), ':', [float(l) for l in test_losses])

    print(f'train loss: {np.mean(train_losses)}, test loss: {np.mean(test_trial_losses)}')

    # plot RNN losses
    plt.figure(figsize=(18, 12))
    for trial_i in range(len(test_trial_losses[0])):
        plt.plot(np.arange(nepoch), [l[trial_i] for l in test_trial_losses],
                 label=f'trial #{test_trials[trial_i]}', alpha=.3)
    plt.plot(train_losses, label='train loss', color='gray', linewidth=8)
    plt.plot([np.mean(l) for l in test_trial_losses], label='test loss', color='red', linewidth=5)

    plt.xlabel('epoch')
    leg = plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.title(f'RNN train={np.mean(train_losses)}, test={np.mean(test_trial_losses)}, k={k}, epoch={nepoch}, '
              f'hidden={hidden_dim}, num layers: {num_layers}, grid: {grid_conv}')
    plt.savefig(f'results/{model_name}.png', bbox_extra_artists=(leg,), bbox_inches='tight')

    # save model
    model_path = f'models/{model_name}.pth'
    torch.save({'epoch': nepoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': np.mean(train_losses), 'test_loss': np.mean(test_trial_losses)}, model_path)
