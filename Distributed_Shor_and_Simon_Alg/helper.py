import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

def qft_circuit(wires):
    """实现量子傅里叶变换(QFT)"""
    n = len(wires)

    # 应用交换门调整比特顺序
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - 1 - i]])

    # 应用受控相位门和Hadamard门
    for j in range(n - 1, -1, -1):
        qml.Hadamard(wires=wires[j])
        for k in range(j - 1, -1, -1):
            # 计算旋转角度：2π/(2^(j-k+1))
            angle = 2 * np.pi / (2 ** (j - k + 1))
            # 应用受控相位门
            qml.ControlledPhaseShift(angle, wires=[wires[k], wires[j]])


def iqft_circuit(wires, shift=0):
    """实现逆量子傅里叶变换(IQFT)"""
    n = len(wires)

    # 应用Hadamard门和受控相位门
    for j in range(n):
        for k in range(j):
            # 计算旋转角度：-2π/(2^(j-k+1))
            angle = -2 * np.pi / (2 ** (j - k + shift + 1))
            # 应用受控相位门
            qml.ControlledPhaseShift(angle, wires=[wires[j], wires[k]])
        qml.Hadamard(wires=wires[j])

    # 应用交换门调整比特顺序
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - 1 - i]])

def dec2bin(dec_, n=0):
    bin_ = bin(dec_)
    bin_ = bin_[2:]
    while len(bin_)<n:
        bin_ = "0"+bin_
    return bin_