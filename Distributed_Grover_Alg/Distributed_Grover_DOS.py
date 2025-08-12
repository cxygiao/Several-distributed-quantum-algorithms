
import math
import matplotlib.pyplot as plt
import random
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile,qasm2
from qiskit_aer import *
from qiskit.circuit.library import MCMTGate, ZGate, RZGate, XGate, UGate, PhaseGate

from qiskit.visualization import plot_histogram

# #本程序实现了单目标分布式精确Grover算法
#
# #主程序（在代码中标注为#Main）
# 实现了针对给定目标应用单目标分布式精确Grover算法在无序数据库中进行搜索，测量输出目标串的目的
# 包含参数设置（节点对应目标态、仿真次数）、图像输出
#
# #Oracle定义（在代码中标注#Definition of Oracle）
# 主要是对算法中所使用到的黑盒Oracle进行定义
#
# #扩散算子定义（在代码中标注#Definition of diffusive operator）
# 主要是对算法中所使用到的扩散算子进行定义
#
# #二进制转换（在代码中标注#Definition of converting to the fixed length binary string）
# 为了让输出的图像显示二进制串等功能特别定义了一个转换二进制固定位数的函数

#Definition of Oracle
def amplitude_oracle(marked_states,varphi):
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    num_qubits = len(marked_states[0])

    qc = QuantumCircuit(num_qubits)
    for target in marked_states:
        rev_target = target[::-1]
        zero_inds = [
            ind
            for ind in range(num_qubits)
            if rev_target.startswith("0", ind)
        ]
        if not zero_inds:
            qc.compose(MCMTGate(PhaseGate(varphi), num_qubits - 1, 1), inplace=True)

        else:
            qc.x(zero_inds)
            qc.compose(MCMTGate(PhaseGate(varphi), num_qubits - 1, 1), inplace=True)
            qc.x(zero_inds)

        qc.barrier()

    return qc

#Definition of diffusive operator
def diffusion_oracle(qc,marked_states,phi):
    num_qubits = len(marked_states[0])

    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.compose(MCMTGate(PhaseGate(phi), num_qubits - 1, 1), inplace=True)

    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc

def convert_to_fixed_length_binary(num, length):
    binary = bin(num)[2:]
    return binary.zfill(length)


#Main
#Setting
marked_states = ["011111"]

num_qubits = len(marked_states[0])
num_iterations=math.floor(
    math.pi*math.sqrt(2**num_qubits) / 4
)

qc = QuantumCircuit(num_qubits)

qc.h(range(num_qubits))
l=1
for l in range(1,num_iterations):
    phi_l=math.pi/2
    varphi_l =math.pi/2
    amp_oracle = amplitude_oracle(marked_states,varphi_l)
    oracle=diffusion_oracle(amp_oracle,marked_states,phi_l)
    qc.compose(oracle,inplace=True)
    l+=1

#Accurate angle
acc_theta=math.asin(1/math.sqrt(num_qubits))
acc_phi=2*math.atan(1/math.sqrt(((math.sin(2*acc_theta)**2)*(math.tan((2*num_iterations+1)*acc_theta)**2))-math.cos(2*acc_theta)**2))
acc_psi=math.atan(1/(math.tan(acc_phi/2)*(-math.cos(2*acc_theta))))
amp_oracle = amplitude_oracle(marked_states, acc_phi)
oracle = diffusion_oracle(amp_oracle, marked_states, acc_psi)
qc.compose(oracle, inplace=True)

qc.measure_all()

#Seeting of Simulation
backend = Aer.get_backend('qasm_simulator')
transpile_qc=transpile(qc,backend=backend)
job = backend.run(transpile_qc,shots=1000,memory=True)
ideal_counts = job.result().get_counts()

plot_histogram(ideal_counts)

plt.show()
