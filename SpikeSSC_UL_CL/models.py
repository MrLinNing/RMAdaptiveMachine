
import argparse, pickle, torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




lens = 0.25


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thresh):
        ctx.save_for_backward(input)
        ctx.vth = thresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        thresh= ctx.vth

        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens), None


def lsm_update(fc1,fc2, x,h, mem, spike, decay = 0.3, thresh=0.3):

    mem = mem * decay * (1 - spike) + fc1(x) + fc2(h)
    spike = act_fun(mem, thresh)
    return mem, spike


act_fun = ActFun.apply

## LSM
class SNN_LSM_Model(nn.Module):
    def __init__(self, cfg = [140, 512, 20], time_window =152,  
                thresh=0.3, decay_lsm=0.9, const = 0.01,
                mean_l1=0.00156, std_l1=6.61329, mean_l2=0.00606, std_l2=6.67268):
        super(SNN_LSM_Model, self).__init__()

        self.cfg = cfg   # network node  [12*12*2,90,10]
        self.tw = time_window
        self.decay_lsm = decay_lsm  # decay
        self.thresh = thresh  # Vth

        self.const = const

        self.mean_l1 = mean_l1
        self.std_l1 = std_l1
        self.mean_l2 = mean_l2
        self.std_l2 = std_l2
        
        self.fc1 = nn.Linear(self.cfg[0], self.cfg[1], bias = True)
        self.fc1h = nn.Linear(self.cfg[1], self.cfg[1], bias = True)

        self.fc2 = nn.Linear(self.cfg[1], self.cfg[-1], bias=True)
        

    def forward(self, input):

        input = input.float()
        
        # print('input shape is',input.shape)
        batch_size = input.shape[0]
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.cfg[1], device=input.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, self.cfg[-1], device=input.device)
        
        # LSM innitail state
        h = torch.zeros(batch_size, self.cfg[1], device=input.device)
        
        for step in range(input.shape[1]):

            x = input[:, step, :].view(batch_size, -1)
            # print('x shape is',x.shape)

            h1_mem, h1_spike = lsm_update(self.fc1, self.fc1h, x, h, h1_mem, h1_spike, self.decay_lsm, self.thresh)
            h = h1_spike
            h1_sumspike = h1_sumspike + h1_spike

        outputs = self.fc2(h1_sumspike/self.tw)
        return outputs


class SNN_LSM_count_Model(nn.Module):
    def __init__(self, cfg = [140, 512, 20], time_window =152,  
                thresh=0.3, decay_lsm=0.9, const = 0.01,
                mean_l1=0.00156, std_l1=6.61329, mean_l2=0.00606, std_l2=6.67268):
        super(SNN_LSM_count_Model, self).__init__()
        self.cfg = cfg   # network node  [12*12*2,90,10]
        self.tw = time_window
        self.decay_lsm = decay_lsm  # decay
        self.thresh = thresh  # Vth

        self.const = const

        self.mean_l1 = mean_l1
        self.std_l1 = std_l1
        self.mean_l2 = mean_l2
        self.std_l2 = std_l2
        
        self.fc1 = nn.Linear(self.cfg[0], self.cfg[1], bias = True)
        self.fc1h = nn.Linear(self.cfg[1], self.cfg[1], bias = True)

        self.fc2 = nn.Linear(self.cfg[1], self.cfg[-1], bias=True)
        
        self.fc1.snn_ops_avg = 0
        self.fc1h.snn_ops_avg = 0
        self.total_snn_ops_avg = 0

        self.fc1.gpu_ops = 0
        self.fc1h.gpu_ops = 0
        self.fc2.gpu_ops = 0
        self.total_gpu_ops = 0

    def count_rram_snn_operations(self,input_spikes, fc_layer):
        # count the number of operations (MACs) for a fully connected layer under a sparse input
        # input_spikes: tensor of shape (batch_size, input_size)
        # fc_layer: nn.Linear layer
        batch_size = input_spikes.size(0)
        input_size = fc_layer.in_features
        output_size = fc_layer.out_features
        
        # Count non-zero input spikes
        non_zero_counts = torch.count_nonzero(input_spikes, dim=1)  # shape: (batch_size,)
        # Total operations is sum of non-zero counts multiplied by output size
        batch_ops = torch.sum(non_zero_counts * output_size).item()
        self.total_snn_ops_avg += batch_ops / batch_size
        fc_layer.snn_ops_avg += batch_ops / batch_size
        return self.total_snn_ops_avg, fc_layer.snn_ops_avg
    
    def count_gpu_operations(self,inputs, fc_layer):
        # count the number of operations (MACs) for a fully connected layer under a sparse input
        # input_spikes: tensor of shape (batch_size, input_size)
        # fc_layer: nn.Linear layer
        batch_size = inputs.size(0)
        input_size = fc_layer.in_features
        output_size = fc_layer.out_features
        
        # Total operations is input size multiplied by output size
        ops = input_size * output_size
        self.total_gpu_ops += ops
        fc_layer.gpu_ops += ops
        return self.total_gpu_ops, fc_layer.gpu_ops


    def forward(self, input):

        input = input.float()
        
        # print('input shape is',input.shape)
        batch_size = input.shape[0]
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.cfg[1], device=input.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, self.cfg[-1], device=input.device)
        
        # LSM innitail state
        h = torch.zeros(batch_size, self.cfg[1], device=input.device)
        
        for step in range(input.shape[1]):

            x = input[:, step, :].view(batch_size, -1)
            # print('x shape is',x.shape)

            self.count_rram_snn_operations(x, self.fc1)
            self.count_rram_snn_operations(h, self.fc1h)
            # print('Average RRAM SNN operations so far:', self.total_snn_ops_avg)
            # print('FC1 layer average RRAM SNN operations so far:', self.fc1.snn_ops_avg)
            # print('FC1h layer average RRAM SNN operations so far:', self.fc1h.snn_ops_avg)
            self.count_gpu_operations(x, self.fc1)
            self.count_gpu_operations(h, self.fc1h)
            # print('Total GPU operations so far:', self.total_gpu_ops)
            # print('FC1 layer GPU operations so far:', self.fc1.gpu_ops)
            # print('FC1h layer GPU operations so far:', self.fc1h.gpu_ops)

            h1_mem, h1_spike = lsm_update(self.fc1, self.fc1h, x, h, h1_mem, h1_spike, self.decay_lsm, self.thresh)
            h = h1_spike
            h1_sumspike = h1_sumspike + h1_spike

        self.count_gpu_operations(h1_sumspike/self.tw, self.fc2)
        # print('Total GPU operations so far:', self.total_gpu_ops)
        # print('FC2 layer GPU operations so far:', self.fc2.gpu_ops)
        outputs = self.fc2(h1_sumspike/self.tw)
        return outputs
