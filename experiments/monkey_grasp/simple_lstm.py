import numpy as np
import sys, os
import torch
from torch.nn import Module, LSTM, Sequential, Linear, MSELoss


class SimpleLSTM(Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        self.lstm = LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=.3)
        self.lin = Linear(hidden_dim, out_dim)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.h, self.c = None, None

    def init_hidden(self, batch_size: int):
        dev = next(self.parameters()).device
        return (torch.rand((self.num_layers, batch_size, self.hidden_dim), device=dev),
               torch.randn((self.num_layers, batch_size, self.hidden_dim), device=dev))

    def forward(self, x):
        x, (self.h, self.c) = self.lstm(x, (self.h, self.c))
        return self.lin(x)

    def tbptt_train(self, trial_gens, loss_fun, optimizer, k: int):
        self.train()
        dev = next(self.parameters()).device
        ntrials = len(trial_gens)  # same as batch size now
        self.h, self.c = self.init_hidden(batch_size=ntrials)

        out_of_seq = False
        losses = []
        while not out_of_seq:
            batch_x, batch_y = [], []
            try:
                for trial_gen in trial_gens:
                    seq = [next(trial_gen) for _ in range(k)]
                    batch_x.append([x for x, y in seq])
                    batch_y.append([y for x, y in seq])

                batch_x = torch.tensor(batch_x, dtype=torch.float32, device=dev)  # ntrial x k x feat,
                batch_y = torch.tensor(batch_y, dtype=torch.float32, device=dev)  # where ntrial == batch

                # actual training
                self.zero_grad()
                self.h.detach_()
                self.c.detach_()
                self.h, self.c = self.h.detach(), self.c.detach()
                y_pred = self.forward(batch_x)
                loss = loss_fun(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu())

            except (RuntimeError, StopIteration):
                print('k is off', file=sys.stderr)  # TODO handle when trial_len % k != 0 for all trials
                break

        return np.mean(losses)

    def test(self, trial_gens, loss_fun):
        dev = next(self.parameters()).device
        with torch.no_grad():
            self.eval()
            self.h, self.c = self.init_hidden(batch_size=1)  # testing each trial separately, no batching
            losses = []
            for trial_gen in trial_gens:
                samples = [sample for sample in trial_gen]
                x = torch.tensor([sample[0] for sample in samples], dtype=torch.float32, device=dev)
                y = torch.tensor([sample[1] for sample in samples], dtype=torch.float32, device=dev)
                x, y = torch.unsqueeze(x, 0), torch.unsqueeze(y, 0)  # add batch dim
                y_pred = self(x)
                losses.append(loss_fun(y_pred, y).cpu())

            return losses
