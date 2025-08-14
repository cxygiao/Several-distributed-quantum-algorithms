# Several-distributed-quantum-algorithms
# 分布式量子算法示例（PennyLane + Qiskit）

本仓库包含 5 个用于教学/实验的量子算法/工具脚本：

- **PennyLane**
  - `Distributed_Shor.py`：整数的**阶发现**（Shor 子程序），含测量分布与连分数后处理。
  - `Distributed_Simon.py`：两节点（\(t=1\)）**分布式 Simon** 原型，含并行 Oracle、量子排序网络 `U_Sort`、模 2 高斯消元与 \(s\) 的重构。
- **Qiskit**
  - `Distributed_Grover_Exp.py`：**分布式 Grover + 计数/幅度估计（QAE 思路）**；使用受控 \(G^{2^i}\) 估计解数的粗略区间，再自适应选择迭代轮次并统计结果。
- **编译/评估工具**
  - `compiler.py`：基于电路划分与线序生成的**传输代价评估**脚本，支持谱聚类划分、穷举/随机线序与**前瞻（look-ahead）**传输代价。

> 以上代码用于原理演示与小规模验证，非针对真实硬件的高性能实现。

---

## 目录结构

```
.
├── Distributed_Shor.py              # Shor 阶发现（相位估计 + 连分数）[PennyLane]
├── Distributed_Simon.py             # 分布式 Simon（并行查询 + 量子排序 + 经典后处理）[PennyLane]
├── Distributed_Grover_Exp.py        # 分布式 Grover + 计数/幅度估计 [Qiskit]
└── Compiler.py                      # 划分与传输代价评估（支持 look-ahead / 谱聚类 / 随机与穷举线序）
```

---

## 运行环境

- Python 3.9+（建议 3.13）
- **PennyLane 栈**：`pennylane`（默认仿真器 `default.qubit`），`matplotlib`
- **Qiskit 栈**：`qiskit`、`qiskit-aer`、`matplotlib`
- **编译/评估工具依赖**：项目内 `Utils/` 与 `partition/` 模块（脚本已按该相对路径导入）


---

## 安装

PennyLane 示例所需：
```bash
pip install pennylane matplotlib
```

Qiskit 示例所需：
```bash
pip install qiskit qiskit-aer matplotlib
```

> 如需显式安装系统 NumPy：
> ```bash
> pip install numpy
> ```

---

## 快速开始

**PennyLane：**

- Shor 阶发现：
  ```bash
  python Distributed_Shor.py
  ```
- 分布式 Simon（脚本内含 `if __name__ == "__main__":` 测试入口）：
  ```bash
  python Distributed_Simon.py
  ```

**Qiskit：**

- 分布式 Grover + 计数/幅度估计：
  ```bash
  python Distributed_Grover_Exp.py
  ```

**编译/评估工具：**

- 计算指定电路在不同划分/线序下的传输代价：
  ```bash
  python compiler.py
  ```

---

## 脚本详解

### 1) `Distributed_Shor.py`（PennyLane）

**目的**：对 \(N=33\) 使用相位估计求 \(a=7\) 的乘法阶 \(r\)。

**关键参数**：
- `N = 33`：待分解的数；
- `a = 7`：与 \(N\) 互质的基数；
- `L = ceil(log2(N)) = 6`（脚本注释处若写 4，实际应为 6），用于第二寄存器位数；
- `t_1, t_2`：两段相位寄存器的位宽（由 `L` 与精度参数 `p` 推导）。

**输出**：
- 终端打印：测量概率分布与候选阶；
- 在有图形后端时，使用 `qml.draw_mpl` 绘制电路图。

---

### 2) `Distributed_Simon.py`（PennyLane）

**目的**：在 \(n=3\)、\(t=1\)（两节点）与输出位数 \(m=2\) 的设定下，演示分布式 Simon：

- 两次并行 Oracle 查询（示例给出 `s=110` 的真值表）；
- 自定义**量子排序网络** `U_Sort` 对两个 2 比特寄存器进行排序；
- 再次查询与干涉，随后对第一寄存器施加 Hadamard 并测量；
- **经典后处理**：依据 \(z \cdot s \equiv 0 \pmod 2\) 构造方程并做模 2 高斯消元，调用 `find_s()` 重构 \(s\)。

**输出**：
- 终端：测量分布与模 2 RREF（行简化阶梯形）；
- `Prob_Simon.pdf`：测量概率直方图；
- 返回值：重构得到的 \(s\)。

**注意事项**：
- 无显示环境请将 `TkAgg` 改为 `Agg` 并使用 `plt.savefig(...)`；
- 量子排序网络为原型实现，未做深度/门数优化；
- 示例中的 `simon_oracle_0/1` 使用硬编码真值表，实际可替换为通用的多控门组合以支持更大规模与任意 \(s\)。

---

### 3) `Distributed_Grover_Exp.py`（Qiskit）— **分布式 Grover + 计数/幅度估计**

**目的**：先对解数进行粗略估计（利用受控 \(G^{2^i}\) 的幅度估计思路），再据此自适应选择 Grover 迭代轮次；脚本循环多次统计直方图，展示搜索效果。

**实现要点（与源码对应）**：
- `GroverOperator(oracle)`：从自定义 Oracle 生成放大算子；
- 取 `con_qubits = ceil(sqrt(n))` 个控制位，构造受控 \(G^{2^i}\) 并测量控制寄存器以推断解数范围；
- 主程序循环多次（源码中 `while count < 500:`），聚合测量结果；
- 同时绘制“全部态直方图”和“过滤零频次后的直方图”。

**关键参数**：
- `marked_states`：**必须**为**字符串列表**，例如 `['1001', '0010']`。
  - ⚠️ 请勿使用中文逗号或把多个比特串写成一个字符串（如 `'1001，0010'`），否则长度解析与计数空间会出错。
- `con_qubits`：控制寄存器位数（源码中为经验选择 `ceil(sqrt(n))`）。

**输出**：
- 控制寄存器估计结果与聚合计数直方图。

**提示**：
- 建议在具体应用中校准 `con_qubits` 与候选迭代轮次，或替换为标准 QAE/Iterative QAE 以提升估计稳定性。

---

### 4) `Compiler.py` — 分布式电路**划分/线序**与**传输代价**评估

**用途**：
- 从 OpenQASM 电路生成门序列，并在给定**分区数/规模**与**线序（line sequence）**的设定下，计算**传输代价**；
- 支持**前瞻（look-ahead）**传输代价模型，与**不前瞻**代价对比；
- 支持**谱聚类**（`spectral_min_cut`）生成分区标签；
- 提供两种线序搜索策略：**穷举**（`generate_line`）与**随机采样**（`random_line_by_qubit`）。

**核心函数**（与源码同名）：
- `count_st_by_partition(partition_method, k, gate_list)`：按划分方法（如 `spectral_clustering`）输出 look-ahead 传输代价；
- `k_count_st_num_ahead(partition_labels, gate_list)`：给定分区标签，生成线序并计算 look-ahead 代价；
- `k_count_min_st_num_ahead_by_iteration(gate_list, cut_list)`：**穷举**所有线序，返回各线序的 look-ahead 代价列表；
- `k_count_min_st_num_ahead_by_random_partition(gate_list, cut_list, random_num)`：**随机**采样线序评估代价；

**输入/依赖**：
- OpenQASM 文件（默认 `./qasm/qasm_czl/mini_alu_305.qasm`，可在脚本顶部修改 `input_filename`）；
- 工程内模块：`Utils/`（如 `generate_line`, `direct_calculation_of_tc_look_ahead` 等）、`partition/`（如 `spectral_min_cut`）；

**关键变量**：
- `cut_list`：各分区规模（如 `[5, 5]`）；
- `circuit_qubit`：由门列表自动推断的量子比特数；
- `random_num`：随机线序采样次数。

**运行**：
```bash
python compiler.py
```
脚本会打印：
- 穷举/随机线序下的**最小传输代价**与进度；
- 可选的全局门统计与不同代价模型（无前瞻 vs. 前瞻）的对比；
- 运行耗时。

**注意**：脚本设置了较高的递归深度（`sys.setrecursionlimit(50000)`），以避免深图遍历时栈溢出。

