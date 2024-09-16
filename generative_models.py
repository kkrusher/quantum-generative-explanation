from abc import abstractmethod

import pennylane as qml

# from pennylane import numpy as np
import numpy as np
import torch
import torch.nn as nn

from oqb.models.quantum.embedding.embedding import build_embedding
from oqb.models.quantum.measurement.measure import build_measurement


class QuantumGenerative(nn.Module):
    def __init__(
        self,
        i_qubits,
        o_qubits,
        a_qubits=0,
        n_layers=10,
        embedding="gate",
        reuploading=1,
        two_qubit_gate_type="CZ",
        **kwargs
    ):
        super().__init__()
        self.i_qubits = i_qubits
        self.o_qubits = o_qubits
        self.a_qubits = a_qubits

        self.all_qubits = self.i_qubits + self.o_qubits + self.a_qubits
        self.n_layers = n_layers
        self.embedding = embedding
        self.reuploading = reuploading

        if two_qubit_gate_type == "CZ":
            self.two_qubit_gate_type = qml.ops.CZ
        elif two_qubit_gate_type == "CNOT":
            self.two_qubit_gate_type = qml.ops.CNOT
        else:
            raise ValueError("Invalid two_qubit_gate_type. Must be 'CZ' or 'CNOT'.")

        # 定义量子设备
        self.dev = qml.device("default.qubit", wires=self.all_qubits)

        # 确定权重的形状
        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.all_qubits
        )

        # 创建一个随机的权重张量，并将其转换为可训练的参数
        self.params = nn.ParameterList(
            [torch.nn.Parameter(torch.rand(shape)) for _ in range(self.reuploading)]
        )
        # 创建一个新的 nn.Parameter 对象，作为最后一层单比特旋转门
        param_last_layer = nn.Parameter(torch.randn(self.all_qubits, 3))
        # 将新参数添加到 param_list 中
        self.params.append(param_last_layer)

    def quantum_circuit(self, params, input):
        @qml.qnode(self.dev, interface="torch")
        def circuit(params, input):
            for i in range(self.reuploading):
                qml.RY(input[0], wires=0)  # J1
                qml.RY(input[1], wires=1)  # J2
                qml.StronglyEntanglingLayers(
                    weights=params[i],
                    wires=range(self.all_qubits),
                    imprimitive=self.two_qubit_gate_type,
                )

                # # todo 考虑换个层
                # build_embedding(
                #     params,
                #     self.all_qubits,
                #     embedding_type="gate",
                #     rotation=self.one_qubit_rotation_type,
                #     two_qubit_gate_type=self.two_qubit_gate_type,
                # )

            # 在整个电路的最后，在每个量子比特上应用一层 Rot 门
            for j in range(self.all_qubits):
                qml.Rot(
                    *params[-1][j], wires=j
                )  # 使用最后一个 reuploading 的最后一组参数

            # return qml.state()
            return qml.density_matrix(
                list(range(self.i_qubits, self.i_qubits + self.o_qubits))
            )

        return circuit(params, input)

    # 使用了circuit之后，不能同时处理一个batch，需要逐个处理，然后stack起来
    def forward(self, inputs):
        # 逐个处理每个样本，并收集结果
        quantum_outputs = [self.quantum_circuit(self.params, x) for x in inputs]
        quantum_output = torch.stack(quantum_outputs)
        return quantum_output
