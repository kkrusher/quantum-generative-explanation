import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import json
import copy
import sys
import pandas as pd
import os
from os import path
import pennylane as qml


# output，target都是密度矩阵(density,density)
def batch_dd_fidelity_loss(output_batch, target_batch):
    losses = []
    for output_state, target_state in zip(output_batch, target_batch):
        inner_product = qml.math.dot(qml.math.conj(output_state), target_state)
        loss = 1 - qml.math.abs(inner_product) ** 2
        # dm_output_state = qml.math.dm_from_state_vector(output_state)
        # dm_target_state = qml.math.dm_from_state_vector(target_state)
        # loss = 1 - qml.math.fidelity(dm_output_state, dm_target_state)
        losses.append(loss)
    return torch.mean(torch.stack(losses))


# output是密度矩阵(density)，target是纯态(pure)，向量
def batch_dp_fidelity_loss(output_batch, target_batch):
    losses = []
    for output_state, target_state in zip(output_batch, target_batch):
        # 计算目标态的密度矩阵
        target_density_matrix = qml.math.outer(
            target_state, qml.math.conj(target_state)
        )
        # print(output_state)
        # print(target_density_matrix)
        # 计算保真度
        fidelity = qml.math.fidelity(output_state, target_density_matrix)
        losses.append(1 - fidelity)

    return torch.mean(torch.stack(losses))


def MSE(input, target):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(input, target)
    # return torch.nn.MSELoss(input, target)


def CrossEntropy(input, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    return cross_entropy_loss(input, target)
