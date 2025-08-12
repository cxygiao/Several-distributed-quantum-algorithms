import pennylane as qml
from pennylane import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 参数设置
N = 13  # 要分解的数
a = 7  # 与N互质的随机整数
eps = 0.5
t = int(np.ceil(np.log2(N)) + np.ceil(np.log2(2+1/(2*eps)))) # 第一寄存器量子比特数 (精度)
L = int(np.ceil(np.log2(N)))  # 第二寄存器量子比特数 (ceil(log2(N)))

# 创建设备
dev = qml.device("default.qubit", wires=t + L)
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
#U2_matrix = build_U_matrix(2)  # U^2: |x> -> |a^2*x mod N>


# 量子电路定义
@qml.qnode(dev)
def shor_order_finding():
    # 初始化第二寄存器为 |1>
    qml.PauliX(wires=t)

    # 第一寄存器应用 Hadamard 门
    for wire in range(t):
        qml.Hadamard(wires=wire)

    wire = [i for i in range(t, t + L)]

    # 控制模幂运算
    # U^{2^0} 受第0位控制
    #qml.ControlledQubitUnitary(U_matrix, wires=[0] + wire)
    # U^{2^1} 受第1位控制
    #qml.ControlledQubitUnitary(U2_matrix, wires=[1] + wire)
    # 更高次幂 (U^{4}, U^{8}) 是单位矩阵，可省略

    for i in range(t):
        power = 2 ** (i)
        qml.ControlledQubitUnitary(
            qml.math.linalg.matrix_power(U_matrix, power),
            #work_wires=[wire],
            wires=[i] + wire
        )

    # 应用逆量子傅里叶变换 (IQFT) 到第一寄存器
    qml.adjoint(qml.QFT)(wires=range(t))

    # 测量第一寄存器
    return qml.probs(wires=range(t))


# 运行电路并打印结果
probs = shor_order_finding()
print("测量概率分布:", probs)

# 提取可能结果
measurements = np.where(probs > 0.001)[0]
print("\n可能的测量值 (相位估计):", measurements)


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
for m in measurements:
    if m == 0:
        continue
    # 相位 = m / 2^t
    phase = m / (2 ** t)

    # 使用连分数展开逼近
    fracs = continued_fractions(m, 2 ** t)
    convergents = []
    for i in range(1, len(fracs) + 1):
        num, den = 0, 1
        for j in range(i - 1, -1, -1):
            num, den = den, fracs[j] * den + num
        convergents.append((den, num))  # 分数 = num/den
    print(convergents)

    # 检查可能的 r
    for denom, _ in convergents:
        r = denom
        if r < N and (a ** r) % N == 1:
            print(f"\n以概率{probs[m]:2f}找到可能的阶 r = {r} (来自相位估计 {m}/{2 ** t} ≈ {phase:.3f})")
            sol_set.append(m)
            break

# Plot the frequency distribution
labels = ['Not target' for i in range(2**t)]
color = ['#3682be' for i in range(2**t)]
for sol in sol_set:
    labels[sol] = 'target'
    color[sol] = '#f05326'

plt.bar([i for i in range(2**t)], probs, color=color)
plt.xlabel('Measurement Result')
plt.ylabel('Probability')
plt.title('Probability of Quantum Measurement Results')
plt.xticks(rotation=0)
plt.show()

fig, ax = qml.draw_mpl(shor_order_finding)()
plt.show()