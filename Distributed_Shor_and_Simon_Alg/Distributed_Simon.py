import pennylane as qml
from pennylane import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from pennylane.operation import Operation
from functools import partial

from helper import dec2bin


# 配置参数
n = 3  # 输入字符串总长度
t = 1  # 分布式节点数指数 (节点数 = 2^t)
num_nodes = 2 ** t  # 实际节点数
m = 2  # 函数输出位数 (简化)


def simon_oracle(input_wires, output_wires):
    """
    严格按照真值表实现的 Simon Oracle
    输入: 3个量子比特 (input_wires[0:3])
    输出: 2个量子比特 (output_wires[0:2])
    s = 110

    真值表:
        f(000) = 00
        f(110) = 00
        f(001) = 01
        f(111) = 01
        f(010) = 10
        f(100) = 10
        f(011) = 11
        f(101) = 11
    """
    x0, x1, x2 = input_wires
    y0, y1 = output_wires

    # 辅助量子比特用于多控制门
    aux = max(output_wires) + 1

    # 实现真值表逻辑

    # f0 (高位) 为 1 的情况: 010,100,011,101
    # 即: (¬x0∧x1) ∨ (x0∧¬x1) = x0 XOR x1

    # 使用 CNOT 实现 XOR
    qml.CNOT(wires=[x0, y0])
    qml.CNOT(wires=[x1, y0])

    # f1 (低位) 为 1 的情况: 001,111,011,101
    # 即: x2 ∧ (¬(x0∧x1)) 但真值表显示更简单: f1 = x2

    # 直接使用 CNOT
    qml.CNOT(wires=[x2, y1])

def simon_oracle_0(input_wires, output_wires):
    """
    严格按照真值表实现的 Simon Oracle
    输入: 3个量子比特 (input_wires[0:3])
    输出: 2个量子比特 (output_wires[0:2])
    s = 110

    真值表:
        f(000) = 00
        f(110) = 00
        f(001) = 01
        f(111) = 01
        f(010) = 10
        f(100) = 10
        f(011) = 11
        f(101) = 11
    """
    x0, x1 = input_wires[0], input_wires[1]
    y0, y1 = output_wires[0], output_wires[1]

    # 辅助量子比特用于多控制门
    aux = max(output_wires) + 1

    # 实现真值表逻辑

    # f0 (高位) 为 1 的情况: 010,100,011,101
    # 即: (¬x0∧x1) ∨ (x0∧¬x1) = x0 XOR x1

    # 使用 CNOT 实现 XOR
    qml.CNOT(wires=[x0, y0])
    qml.CNOT(wires=[x1, y0])

    # f1 (低位) 为 1 的情况: 001,111,011,101
    # 即: x2 ∧ (¬(x0∧x1)) 但真值表显示更简单: f1 = x2

    # 直接使用 CNOT
    #qml.CNOT(wires=[x2, y1])

def simon_oracle_1(input_wires, output_wires):
    """
    严格按照真值表实现的 Simon Oracle
    输入: 3个量子比特 (input_wires[0:3])
    输出: 2个量子比特 (output_wires[0:2])
    s = 110

    真值表:
        f(000) = 00
        f(110) = 00
        f(001) = 01
        f(111) = 01
        f(010) = 10
        f(100) = 10
        f(011) = 11
        f(101) = 11
    """
    x0, x1 = input_wires
    y0, y1 = output_wires

    # 辅助量子比特用于多控制门
    aux = max(output_wires) + 1

    # 实现真值表逻辑

    # f0 (高位) 为 1 的情况: 010,100,011,101
    # 即: (¬x0∧x1) ∨ (x0∧¬x1) = x0 XOR x1

    # 使用 CNOT 实现 XOR
    qml.CNOT(wires=[x0, y0])
    qml.CNOT(wires=[x1, y0])

    # f1 (低位) 为 1 的情况: 001,111,011,101
    # 即: x2 ∧ (¬(x0∧x1)) 但真值表显示更简单: f1 = x2

    # 直接使用 CNOT
    qml.X(wires=[y1])


def C10X(wires):
    q0, q1, q2 = wires
    qml.PauliX(wires=q1)            # 把 q1 的 0-控制转换为 1-控制
    qml.Toffoli(wires=[q0, q1, q2]) # 现在是 q0==1 且 (原)q1==0 时触发
    qml.PauliX(wires=q1)            # 还原 q1

def OR_ControlledX(wires):
    q0, q1, q2 = wires
    # 情况1: q0=1 时翻转 q2
    qml.CNOT(wires=[q0, q2])
    # 情况2: q0=0 且 q1=1 时翻转 q2
    qml.PauliX(wires=q0)             # 把反控 q0 转成正控
    qml.Toffoli(wires=[q0, q1, q2])  # 现在是 q0'=1 且 q1=1 时触发
    qml.PauliX(wires=q0)             # 还原 q0

def Flip_on_1_or_001(wires):
    q0, q1, q2, q3 = wires
    # 条件1：q0=1 时翻转 q3
    qml.CNOT(wires=[q0, q3])

    # 条件2：q0=0 且 q1=0 且 q2=1 时翻转 q3
    # 先把对 q0、q1 的“0 控制”转成“1 控制”
    qml.PauliX(wires=q0)
    qml.PauliX(wires=q1)
    # 现在触发条件为 q0'=1, q1'=1, q2=1
    qml.MultiControlledX(control_values=[q0, q1, q2], wires=[q0, q1, q2, q3])
    # 还原
    qml.PauliX(wires=q1)
    qml.PauliX(wires=q0)

def copy_register(source_wires, target_wires):
    """将源寄存器的值复制到目标寄存器"""
    for i in range(len(source_wires)):
        qml.CNOT(wires=[source_wires[i], target_wires[i]])


def swap_if_greater(c_wires, d_wires, control_wire):
    """如果控制位为1，则交换两个寄存器"""
    # 交换低位比特
    qml.CSWAP(wires=[control_wire, c_wires[0], d_wires[0]])
    # 交换高位比特
    qml.CSWAP(wires=[control_wire, c_wires[1], d_wires[1]])


def sort_registers(a_wires, b_wires, c_wires, d_wires, anc_wire):
    """比较A和B，并将较小值存入C，较大值存入D"""
    # 1. 复制A->C, B->D
    copy_register(a_wires, c_wires)
    copy_register(b_wires, d_wires)

    # 2. 使用比较器：如果A≥B则设置辅助位为1
    # 高位是否大
    C10X(wires=[a_wires[0], b_wires[0], anc_wire[0]])

    #高位是否小
    C10X(wires=[b_wires[0], a_wires[0], anc_wire[1]])

    #低位是否大
    C10X(wires=[a_wires[1], b_wires[1], anc_wire[2]])

    Flip_on_1_or_001(wires=[anc_wire[0], anc_wire[1], anc_wire[2], anc_wire[3]])

    # 3. 如果A≥B (辅助位=1)，则交换C和D
    swap_if_greater(c_wires, d_wires, anc_wire[3])

    #还原
    Flip_on_1_or_001(wires=[anc_wire[0], anc_wire[1], anc_wire[2], anc_wire[3]])
    C10X(wires=[a_wires[1], b_wires[1], anc_wire[2]])
    C10X(wires=[b_wires[0], a_wires[0], anc_wire[1]])
    C10X(wires=[a_wires[0], b_wires[0], anc_wire[0]])


# 创建分布式Simon算法量子电路
def distributed_simon_circuit(t, n):
    num_nodes = 2 ** t
    input_size = n - t  # 输入寄存器大小
    anx = 4
    total_wires = input_size + num_nodes * m + num_nodes * m + anx  # 总量子比特数

    dev = qml.device("default.qubit", wires=total_wires)

    @qml.qnode(dev)
    def circuit():
        # ==== 算法步骤1: 初始化 ====
        # 第一寄存器 (n-t qubits)
        for wire in range(input_size):
            qml.Hadamard(wires=wire)

        # ==== 算法步骤2: 并行查询Oracle ====
        # 输入寄存器位置
        input_wires = list(range(input_size))

        # 输出寄存器位置
        output_start_0 = input_size
        output_start_1 = input_size + m
        output_wires_0 = list(range(output_start_0, output_start_1))
        output_wires_1 = list(range(output_start_1, output_start_1+m))
        output_wires =  list(range(output_start_0, output_start_0+num_nodes * m))

        # 创建Oracle并应用
        simon_oracle_0(input_wires, output_wires_0)
        simon_oracle_1(input_wires, output_wires_1)

        # ==== 算法步骤3: 量子排序网络 (U_Sort) ====

        # 排序目标寄存器
        sort_start = input_size + num_nodes * m
        sort_wires_0 = list(range(sort_start, sort_start+m))
        sort_wires_1 = list(range(sort_start+m, sort_start + num_nodes * m))

        anx_start = sort_start + num_nodes * m
        anx_wires = list(range(anx_start, anx_start+anx))

        # 应用U_Sort门 (论文中关键创新)
        sort_registers(output_wires_0, output_wires_1, sort_wires_0, sort_wires_1, anx_wires)

        # ==== 算法步骤4: 再次查询Oracle ====
        simon_oracle_1(input_wires, output_wires_1)
        simon_oracle_0(input_wires, output_wires_0)

        # ==== 算法步骤5: 逆傅里叶变换 ====
        for wire in range(input_size):
            qml.Hadamard(wires=wire)

        # ==== 算法步骤6: 测量第一寄存器 ====
        return qml.probs(wires=input_wires)

    return circuit


# 经典后处理 (高斯消元)
def classical_postprocessing(measurements):
    """从测量结果提取s1的正交空间"""
    if not measurements:
        return []

    # 构建方程系统
    equations = []
    for state in measurements:
        if np.any(state):
            equations.append(state)

    if not equations:
        return []

    A = np.array(equations)
    rows, cols = A.shape
    pivot = 0

    # 高斯消元 (模2)
    for col in range(cols):
        # 寻找主元
        pivot_row = -1
        for r in range(pivot, rows):
            if A[r, col] == 1:
                pivot_row = r
                break

        if pivot_row == -1:
            continue  # 该列无主元

        # 交换行
        if pivot_row != pivot:
            A[[pivot, pivot_row]] = A[[pivot_row, pivot]]

        # 消元
        for r in range(rows):
            if r != pivot and A[r, col] == 1:
                A[r] = (A[r] + A[pivot]) % 2

        pivot += 1

    # 提取线性无关的行
    independent_eqs = A[:pivot]

    # 求解零空间
    null_space = []
    for i in range(cols):
        # 构建标准基向量
        vec = np.zeros(cols, dtype=int)
        vec[i] = 1

        # 减去投影
        for j in range(pivot):
            if independent_eqs[j, i] == 1:
                vec = (vec + independent_eqs[j]) % 2

        null_space.append(vec)

    return null_space


# 算法3: 求解s2
def find_s(s1):
    """分布式算法求解s2"""
    f = {'000': 0, '110': 0, '001': 1, '111' : 1, '010': 2, '100': 2, '011':3, '101':3}

    # 步骤1: 查询f(0^{n-t}w) 对所有w
    f_0w = {}
    for w in range(2**t):
        key = '0'*(n-t) + f'{w}'
        f_0w[f'{w}'] = f[key]

    s = ''
    for _ in s1:
        s = s + f'{_}'

    # 步骤2: 查询f(s1, 0^t)
    key = f'{s}' + '0'*t
    f_s10t = f[key]

    # 步骤3: 寻找匹配的v
    for f_key,f_val in f_0w.items():
        if f_val == f_s10t:
            s = s + f_key
            return s

    return None


# 主执行函数
def run_distributed_simon(s, n, t):

    # 步骤2: 运行量子部分 (算法2)
    quantum_circuit = distributed_simon_circuit( t, n)
    probs = quantum_circuit()

    print("输出概率：",probs)


    plt.bar([f'{dec2bin(i, 2)}' for i in range(2 ** 2)], probs)
    plt.xlabel('Measurement Result')
    plt.ylabel('Probability')
    plt.title('Probability of Quantum Measurement Results')
    plt.xticks(rotation=0)
    plt.savefig("Prob_Simon.pdf")

    # 步骤3: 处理量子测量结果
    measurements = []
    num_qubits = n - t

    # 收集显著测量结果
    for i in range(2 ** num_qubits):
        if probs[i] > 0.01:  # 过滤显著测量结果
            bin_str = bin(i)[2:].zfill(num_qubits)
            measurements.append(np.array([int(b) for b in bin_str]))

    # 构建方程矩阵
    matrix = []
    for z in measurements:
        # 忽略全零测量结果
        if np.any(z):
            matrix.append(z)
            print(f"方程: {z} · s = 0 mod 2")

    # 步骤4: 经典后处理 (高斯消元)
    if matrix:
        A = np.vstack(matrix)

        # 高斯消元法 (模2)
        def gauss_elimination_mod2(A):
            A = A.copy()
            rows, cols = A.shape
            pivot = 0

            for col in range(cols):
                # 找到主元
                for r in range(pivot, rows):
                    if A[r, col] == 1:
                        # 交换行
                        if r != pivot:
                            A[[pivot, r]] = A[[r, pivot]]
                        # 消元
                        for r2 in range(pivot + 1, rows):
                            if A[r2, col] == 1:
                                A[r2] = (A[r2] + A[pivot]) % 2
                        pivot += 1
                        break
            return A

        rref_A = gauss_elimination_mod2(A)
        print("方程组的行简化阶梯形式 (模2):")
        print(rref_A)

    # 步骤6: 运行算法3求解s2
    s_final = find_s(rref_A[0])
    return s_final


# 测试函数
def test_distributed_simon():
    s = "110"
    """测试分布式Simon算法"""
    print("\n===== 分布式Simon算法测试 =====")
    print(f"隐藏字符串 s = {s}")
    print(f"参数配置: n={n}, t={t}, 节点数={2 ** t}")

    # 运行算法
    result = run_distributed_simon(s, n, t)

    # 验证结果
    print(f"算法结果: {result}")
    print(f"结果正确: {np.array_equal(s, result)}")
    print("=" * 40)


# 运行测试
if __name__ == "__main__":
    # 小规模测试
    n = 3
    t = 1
    test_distributed_simon()