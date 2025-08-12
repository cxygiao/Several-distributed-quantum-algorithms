import pennylane as qml
from pennylane import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# 设置参数
n = 3  # 输入量子比特数 (隐藏串 s 的长度)
s = np.array([1, 0, 1])  # 隐藏的二进制串 (示例: 101)


# 创建 Simon 算法的 Oracle
class SimonOracle(qml.operation.Operation):
    num_wires = 6  # 总量子比特数: n (输入) + n (输出)
    grad_method = None  # 这个操作不参与梯度计算

    def __init__(self, s, wires=None):
        # 将 s 存储为实例变量
        self.s = s
        super().__init__(wires=wires)

    @property
    def num_params(self):
        return 0  # 无参数

    def compute_decomposition(self, wires):
        # 分解为基本门操作
        s = self.s  # 从实例变量获取 s
        x_wires = wires[:len(s)]  # 输入寄存器
        y_wires = wires[len(s):]  # 输出寄存器
        ops = []

        # 实现 f(x) = f(x⊕s) 的函数
        # 这里使用线性函数: f(x) = (x·a) mod 2，其中 a 是随机向量
        # 但保证 f(x) = f(x⊕s) 当且仅当 a·s = 0 mod 2
        a = np.array([1, 0, 1])  # 固定 a 满足 a·s = 0 (1*1 + 0*0 + 1*1 = 2 ≡ 0 mod 2)

        # 计算 f(x) = x·a mod 2
        for i in range(len(a)):
            if a[i] == 1:
                # 对每个 a[i]=1 的位，添加 CNOT 到输出寄存器
                ops.append(qml.CNOT(wires=[x_wires[i], y_wires[0]]))

        # 添加额外的操作来确保 f(x) = f(x⊕s)
        for i in range(len(s)):
            if s[i] == 1:
                # 添加控制门体现 s 的影响
                for j in range(len(a)):
                    if a[j] == 1:
                        ops.append(qml.CNOT(wires=[x_wires[i], y_wires[j]]))
        return ops


# 创建设备 (n 输入 + n 输出)
dev = qml.device("default.qubit", wires=2 * n)


@qml.qnode(dev)
def simon_algorithm():
    # 初始化输出寄存器为 |0>
    # 输入寄存器: 应用 Hadamard 门创建叠加态
    for i in range(n):
        qml.Hadamard(wires=i)

    # 应用 Simon Oracle
    SimonOracle(s=s, wires=range(2 * n))

    # 再次在输入寄存器应用 Hadamard 门
    for i in range(n):
        qml.Hadamard(wires=i)

    # 测量输入寄存器
    return qml.probs(wires=range(n))


# 运行算法
probs = simon_algorithm()

fig, ax = qml.draw_mpl(simon_algorithm)()
plt.show()

# 可视化概率分布
plt.bar(range(2 ** n), probs)
plt.xlabel('State')
plt.ylabel('Probability')
plt.title(f'Simon Algorithm Results (Hidden s = {s})')
plt.xticks(range(2 ** n), [bin(i)[2:].zfill(n) for i in range(2 ** n)])
plt.show()

# 经典后处理：找出满足 z·s = 0 mod 2 的 z
non_zero_probs = np.where(probs > 0.01)[0]
measurements = [np.array([int(b) for b in bin(i)[2:].zfill(n)]) for i in non_zero_probs]

print("测量结果 (概率>1%):")
for i, z in zip(non_zero_probs, measurements):
    print(f"状态 {bin(i)[2:].zfill(n)}: 概率 = {probs[i]:.4f}")

# 构建方程矩阵
matrix = []
for z in measurements:
    # 忽略全零测量结果
    if np.any(z):
        matrix.append(z)
        print(f"方程: {z} · s = 0 mod 2")

# 解线性方程组找出 s
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


    # 简化行阶梯形式
    rref_A = gauss_elimination_mod2(A)
    print("\n方程组的行简化阶梯形式 (模2):")
    print(rref_A)

    # 找出零空间 (满足 A·s=0 的 s)
    # 这里我们直接验证候选解
    candidates = []
    for test_s in [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]]:
        valid = True
        for row in A:
            if np.dot(row, test_s) % 2 != 0:
                valid = False
                break
        if valid:
            candidates.append(test_s)

    print("\n满足所有方程的候选 s:")
    for c in candidates:
        print(c)

    # 排除零解
    non_zero_candidates = [c for c in candidates if np.any(c)]
    if non_zero_candidates:
        print(f"\n找到隐藏串 s = {non_zero_candidates[0]}")
    else:
        print("\n未找到非零解，函数可能是一对一的 (s=0)")
else:
    print("没有足够的测量结果来求解")