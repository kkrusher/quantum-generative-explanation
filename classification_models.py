from abc import abstractmethod

import pennylane as qml

# from pennylane import numpy as np
import numpy as np
import torch
import torch.nn as nn

from generative_models import *
from oqb.tools import utils


class QuantumClassification(QuantumGenerative):
    def __init__(
        self,
        i_qubits,
        o_qubits,
        a_qubits=0,
        n_layers=10,
        n_classification_layers=10,
        embedding="gate",
        reuploading=1,
        two_qubit_gate_type="CZ",
        categories=2,
        **kwargs,
    ):
        super(QuantumClassification, self).__init__(
            i_qubits,
            o_qubits,
            a_qubits=a_qubits,
            n_layers=n_layers,
            embedding=embedding,
            reuploading=reuploading,
            two_qubit_gate_type=two_qubit_gate_type,
            **kwargs,
        )

        self.observables = None
        self.categories = categories

        self.n_classification_layers = n_classification_layers

        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_classification_layers, n_wires=self.all_qubits
        )
        param_first_layer = torch.nn.Parameter(torch.rand(shape))
        # 创建一个新的 nn.Parameter 对象，作为最后一层单比特旋转门
        param_last_layer = nn.Parameter(torch.randn(self.all_qubits, 3))
        # 将新参数添加到 param_list 中
        self.classification_params = nn.ParameterList(
            [param_first_layer, param_last_layer]
        )

        observables_list = utils.obs(self.categories, self.all_qubits)
        self.set_observables(observables_list)

    def set_observables(self, observables):
        self.observables = observables

    def classification_quantum_circuit(self, params, classification_params, input):
        @qml.qnode(self.dev, interface="torch")
        def classification_circuit(params, classification_params, input):
            for i in range(self.reuploading):
                qml.RY(input[0], wires=0)  # J1
                qml.RY(input[1], wires=1)  # J2
                qml.StronglyEntanglingLayers(
                    weights=params[i],
                    wires=range(self.all_qubits),
                    imprimitive=self.two_qubit_gate_type,
                )

            # 在解码器的最后，在每个量子比特上应用一层 Rot 门
            for j in range(self.all_qubits):
                qml.Rot(
                    *params[-1][j], wires=j
                )  # 使用最后一个 reuploading 的最后一组参数

            # 分类器
            qml.StronglyEntanglingLayers(
                weights=classification_params[0],
                wires=range(self.all_qubits),
                imprimitive=self.two_qubit_gate_type,
            )

            # 在分类器的最后，在每个量子比特上应用一层 Rot 门
            for j in range(self.all_qubits):
                qml.Rot(
                    *classification_params[-1][j], wires=j
                )  # 使用最后一个 reuploading 的最后一组参数

            # # 1.这个只能用在二分类上
            # return qml.probs(wires=[0])

            # 1.这个只能用在四分类上
            return qml.probs(wires=[0, 1])

            # # 2.这种方法测量全部qubit的前category个计算基态，当全部qubit较多时，结果非常小，没法保证归一性
            # return [
            #     build_measurement(obs=obs, observe_state=False)
            #     for obs in self.observables
            # ]

            # 3.这个方法没法保证归一性
            # return [qml.expval(qml.PauliZ(i)) for i in range(self.categories)]

        # 2.3.使用
        # return torch.stack(classification_circuit(params, classification_params, input))

        # 1 使用
        return classification_circuit(params, classification_params, input)

    def forward(self, inputs):
        # # Ensure inputs are tensors with requires_grad=True
        # inputs = [
        #     (
        #         input.clone().detach().requires_grad_(True)
        #         if not input.requires_grad
        #         else input
        #     )
        #     for input in inputs
        # ]

        # Process each sample individually and collect the results
        quantum_outputs = [
            self.classification_quantum_circuit(
                self.params, self.classification_params, x
            )
            for x in inputs
        ]
        quantum_output = torch.stack(quantum_outputs)

        return quantum_output

    # # 使用了circuit之后，不能同时处理一个batch，需要逐个处理，然后stack起来
    # def forward(self, inputs):
    #     # 逐个处理每个样本，并收集结果
    #     quantum_outputs = [
    #         self.classification_quantum_circuit(
    #             self.params, self.classification_params, x
    #         )
    #         for x in inputs
    #     ]
    #     quantum_output = torch.stack(quantum_outputs)
    #     return quantum_output

    def load_generative(self, file_path, freeze_generative=True):
        # 加载模型参数
        loaded_params = torch.load(file_path)

        # 检查加载的参数数量是否与 self.params 相同
        if len(loaded_params) != len(self.params):
            raise ValueError(
                "Loaded parameters length does not match self.params length."
            )

        # 逐个替换 self.params 中的参数
        for i, param_key in enumerate(loaded_params.keys()):
            if loaded_params[param_key].shape != self.params[i].shape:
                raise ValueError(
                    f"Loaded parameter '{param_key}' shape {loaded_params[param_key].shape} does not match self.params[{i}] shape {self.params[i].shape}."
                )

            self.params[i].data = loaded_params[param_key].data

            if freeze_generative:
                self.params[i].requires_grad_(False)

        # 打印初始的 self.params 的前两个元素
        print("Initial self.params (first two elements):")
        for param in self.params:
            print(param[:2])

    def load_generative_and_classification(self, file_path):
        self.load_state_dict(torch.load(file_path))
        # # 加载模型参数
        # loaded_params = torch.load(file_path)

        # # 检查加载的参数数量是否与 self.params 相同
        # if len(loaded_params) != len(self.classification_params):
        #     raise ValueError(
        #         "Loaded parameters length does not match self.classification_params length."
        #     )

        # # 逐个替换 self.params 中的参数
        # for i, param_key in enumerate(loaded_params.keys()):
        #     if loaded_params[param_key].shape != self.classification_params[i].shape:
        #         raise ValueError(
        #             f"Loaded parameter '{param_key}' shape {loaded_params[param_key].shape} does not match self.classification_params[{i}] shape {self.classification_params[i].shape}."
        #         )

        #     self.classification_params[i].data = loaded_params[param_key].data
        #     self.classification_params[i].requires_grad_(False)

        # # 检查加载的参数数量是否与 self.classification_params 相同
        # if (
        #     loaded_params["classification_params"].shape
        #     != self.classification_params.shape
        # ):
        #     raise ValueError(
        #         "Loaded classification parameters shape does not match self.classification_params shape."
        #     )

        # # 替换 self.classification_params 中的参数
        # self.classification_params.data = loaded_params["classification_params"].data
        # self.classification_params.requires_grad_(False)

        # # 打印加载的分类参数
        # print("Loaded classification_params:")
        # print(self.classification_params)
