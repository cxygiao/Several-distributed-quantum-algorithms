#本程序实现了分布式Grover算法

#主程序（在代码中标注为#Main）
实现了针对给定目标应用分布式Grover算法在无序数据库中进行搜索，测量输出目标串的目的
包含参数设置（目标态# Set target_states、y_i长度# The length of y_i、算法循环次数# Setting of shots）、结果以字典形式输出、结果图像输出（图像输出分为包含所有量子态的输出及仅含有有响应状态的输出）

#算子定义（在代码中标注# Definition of operators）
主要是对算法中所使用到的黑盒Oracle进行定义

#算法整体定义（在代码中标注 # Definition of Algorithm）
是对算法的整体封装，包含了算子的调用，两个量子过程中间确定迭代范围的经典过程的编写，以及小规模Grover算法的迭代过程

#二进制转换定义（在代码中标注#Definition of converting to the fixed length binary string）
为了让输出的图像显示二进制串，以及中间过程中的二进制读取转化为十进制等功能特别定义了一个转换二进制固定位数的函数

#随机输出定义（在代码中标注#Definition of random output）
为了满足程序中随机输出固定位数比特串

import math
import matplotlib.pyplot as plt
import random

from numpy.matlib import empty
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister,transpile
from qiskit_aer import *
from qiskit.circuit.library import GroverOperator, MCMTGate, ZGate, RZGate, XGate, UGate
from qiskit.visualization import plot_histogram

# Definition of operators
def grover_oracle(marked_states):

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
            qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
        else:
            qc.x(zero_inds)
            qc.compose(MCMTGate(ZGate(), num_qubits - 1, 1), inplace=True)
            qc.x(zero_inds)

    return qc



# Definition of Algorithm
def DQAA_QAE():

    oracle = grover_oracle(marked_states)

    grover_op = GroverOperator(oracle)
    con_qubits=math.ceil(math.sqrt(num_qubits))

    qubits_sum=grover_op.num_qubits+con_qubits

    qc = QuantumCircuit(qubits_sum,con_qubits)
    qc.h(range(qubits_sum))
    qc.barrier()

    for i in range(con_qubits):
        iteration_operators=QuantumCircuit(num_qubits)
        iteration_operators.compose(grover_op.power(pow(2,i)), inplace=True)
        con_oracle=iteration_operators.to_gate().control(1)
        temp = [i]
        for j in range(con_qubits,qubits_sum):
            temp.append(j)
        qc.append(con_oracle, temp)

    qc.barrier()
    qc.h(range(con_qubits))
    qc.barrier()
    for i in range(con_qubits):
        qc.measure(i,con_qubits-1-i)

    backend = Aer.get_backend('qasm_simulator')
    transpile_qc=transpile(qc,backend=backend)

    job = backend.run(transpile_qc,shots=1,memory=True)
    ideal_counts = job.result().get_memory()


    ideal_counts_binary=ideal_counts[0]
    ideal_counts_decimal=int(ideal_counts_binary,2)
    estimate_decimal=(math.sin(math.pi*ideal_counts_decimal/(2**con_qubits)))**2
    solution_decimal=estimate_decimal*(2**num_qubits)

    round_solution_decimal=round(solution_decimal,0)

    # Distributed Grover algorithm with quantum counting

    # Range of estimate
    round_solution_decimal_min=math.floor(round_solution_decimal-2*math.pi*math.sqrt(round_solution_decimal)-11)
    min_round_solution_decimal=max(round_solution_decimal_min,1)
    round_solution_decimal_max=math.ceil(round_solution_decimal+2*math.pi*math.sqrt(round_solution_decimal)+11)
    max_round_solution_decimal=min(round_solution_decimal_max,2**grover_op.num_qubits)

    # Iteration
    solution_number=min_round_solution_decimal
    while solution_number <= max_round_solution_decimal:

        # Grover algorithm
        optimal_num_iterations = math.floor(math.pi / (4 * math.asin(math.sqrt(solution_number /(2 ** grover_op.num_qubits)))))
        oracle = grover_oracle(marked_states)
        grover_op = GroverOperator(oracle)
        qc = QuantumCircuit(grover_op.num_qubits)
        qc.h(range(grover_op.num_qubits))
        qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
        qc.measure_all()


        backend = Aer.get_backend('qasm_simulator')
        transpile_qc = transpile(qc, backend=backend)
        job = backend.run(transpile_qc, shots=1, memory=True)
        ideal_counts = job.result().get_memory()


        if ideal_counts[0] in marked_states:

            break
        elif solution_number==max_round_solution_decimal:
            ideal_counts_decimal=random.randint(0,(2**grover_op.num_qubits)-1)
            ideal_counts[0]=bin(ideal_counts_decimal)

            break
        else:
            solution_number=solution_number+1

    ideal_counts = job.result().get_counts()

    return ideal_counts

#Definition of converting to the fixed length binary string
def convert_to_fixed_length_binary(num, length):
    binary = bin(num)[2:]
    return binary.zfill(length)

#Definition of random output
def sample_nbit_counts(n, shots) :
    fmt = f"0{n}b"
    counts: dict[str, int] = {}
    for _ in range(shots):
        s = format(random.getrandbits(n), fmt)
        counts[s] = counts.get(s, 0) + 1
    return counts

# Main
# Set target_states
target_states=['101000','100111','001011']
# The length of y_i
y_i_len=2
# Setting of shots
shots=500

for node in range(0,2**y_i_len):
    tail = format(node, f"0{y_i_len}b")
    marked_states = [s[:-y_i_len] for s in target_states if len(s) >= y_i_len and s[-y_i_len:] == tail]
    if len(marked_states)==0:
        count_sum = sample_nbit_counts(len(target_states[0])-y_i_len, shots)
    else:
        num_qubits = len(marked_states[0])
        count=0
        count_sum={}
        for i in range(2**len(marked_states[0])):
            x = convert_to_fixed_length_binary(i,len(marked_states[0]))
            count_sum[x] = 0
        while count< shots:
            ideal_counts_result = DQAA_QAE()
            for result_count, frequent in ideal_counts_result.items():
                count_sum[result_count] += 1
            count += 1

    print("node",node,"=",count_sum)
    plot_histogram(count_sum)
    filtered_data={k: v for k, v in count_sum.items() if v!=0}
    plot_histogram(filtered_data)

plt.show()
