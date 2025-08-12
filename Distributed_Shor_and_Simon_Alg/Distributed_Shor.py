import pennylane as qml
from pennylane import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from helper import dec2bin

# 参数设置
N = 33  # 要分解的数
a = 7  # 与N互质的随机整数
eps = 0.5
p = int(np.ceil(np.log2(2+1/eps))) #p=2

t = int(np.ceil(np.log2(N))) + 1  # 第一寄存器量子比特数 (精度)
L = int(np.ceil(np.log2(N)))  # 第二寄存器量子比特数 (ceil(log2(N))) = 4

t_1 = int(L / 2 + 1 + p)
t_2 = int((3 * L) / 2 + 2 + p)


# 构建模乘算子 U^{2^k} 的矩阵
def build_U_matrix(exponent):
    U = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for x in range(2 ** L):
        if x < N:
            # 计算 a^{exponent} * x mod N
            result = (a ** exponent) * x % N
        else:
            result = x  # 保持未使用的状态不变
        # 设置置换矩阵
        U[result, x] = 1
    return U

# 构建 U 和 U^2 的矩阵
U_matrix = build_U_matrix(1)  # U: |x> -> |a*x mod N>
U2_matrix = build_U_matrix(int(np.ceil(2 ** ((L / 2) - 1))))  # U^2: |x> -> |a^2*x mod N>
# 创建设备
dev = qml.device("default.qubit", wires=t_1 + t_2 + L)

# 量子电路定义
@qml.qnode(dev)
def shor_order_finding():
    # 初始化第二寄存器为 |1>
    qml.PauliX(wires=t_1 + t_2)

    # 第一寄存器应用 Hadamard 门
    for wire in range(t_1+t_2):
        qml.Hadamard(wires=wire)

    wire = [i for i in range(t_1 + t_2, t_1 + t_2 + L)]

    for i, j in enumerate(range(0, t_1)):
        power = 2 ** (i)
        qml.ControlledQubitUnitary(
            qml.math.linalg.matrix_power(U_matrix, power),
            #work_wires=[wire],
            wires=[j] + wire
        )

    # 应用逆量子傅里叶变换 (IQFT) 到第一寄存器
    qml.adjoint(qml.QFT)(wires=range(0, t_1))

    for i, j in enumerate(range(t_1, t_1+t_2)):
        power = 2 ** (i)
        qml.ControlledQubitUnitary(
            qml.math.linalg.matrix_power(U2_matrix, power),
            #work_wires=[wire],
            wires=[j] + wire
        )

    qml.adjoint(qml.QFT)(wires=range(t_1, t_1 + t_2))

    # 测量第一寄存器
    return qml.probs(wires=range(0, t_1 + t_2)), qml.probs(wires=range(0, t_1)), qml.probs(wires=range(0, t_2))


# 运行电路并打印结果
probs, _, _ = shor_order_finding()
print("测量概率分布:", probs)

# 提取可能结果
measurements = np.where(probs > 0.001)[0]
measurements_c = []
for meas in measurements:
    meas_bi = dec2bin(meas, t_1+t_2)

    m_1 = meas_bi[0:t_1]
    m_2 = meas_bi[t_1:t_1 +t_2]

    m_1_first = m_1[0:int(L/2+1)]
    m_1_first_int = int(m_1_first,2)

    m_1_last2 = m_1[int(L/2-1) : int(L/2+1)]
    m_1_last2_int = int(m_1_last2,2)

    m_2_last = m_2[2:t_2]

    m_2_first_2 = m_2[0:2]
    m_2_first_2_int = int(m_2_first_2, 2)

    for corr in [-1, 0 ,1, 2]:
        if (m_1_last2_int + corr) % 4 == m_2_first_2_int:
            m_prefix = (m_1_first_int + corr) % ( 2 ** ((L/2)+1) )
            m = dec2bin(int(m_prefix), ((L/2)+1)) + m_2_last
            break


    # m = dec2bin(int(m_1_first_int), ((L/2)+1)) + m_2_last

    measurements_c.append([meas, int(m,2)])

# 连分数展开寻找r
def continued_fractions(x, denom):
    # 计算 x/denom 的连分数展开
    seq = []
    while x > 0:
        seq.append(denom // x)
        denom, x = x, denom % x
    return seq


# 从测量结果推断阶 r

sol_set = []
is_sol = [0 for _ in range(len(measurements_c))]
for k, (probs_mark, m) in enumerate(measurements_c):

    if m == 0:
        continue
    # 相位 = m / 2^t
    phase = m / (2 ** (2*L+1+p))
    #print(m)


    # 使用连分数展开逼近
    fracs = continued_fractions(m, 2 ** (2*L+1+p))
    convergents = []
    for i in range(1, len(fracs) + 1):
        num, den = 0, 1
        for j in range(i - 1, -1, -1):
            num, den = den, fracs[j] * den + num
        convergents.append((den, num))  # 分数 = num/den

    #print(convergents)

    # 检查可能的 r
    for denom, _ in convergents:
        r = denom
        if r < N and (a ** r) % N == 1:
            print(f"\n以概率{probs[probs_mark]:4f}找到可能的阶 r = {r} (来自相位估计 {m}/{2 ** (2*L+1+p)} ≈ {phase:.3f})")
            sol_set.append(probs_mark)
            break


fig, ax = qml.draw_mpl(shor_order_finding)()
plt.show()