# EMCF 解缠的原理

本页从"空间 × 时间"的角度解释 EMCF（extended minimum cost flow，Pepe & Lanari 2006）解缠。先写出把所有点、所有影像
放在一起解缠的理想问题，再说明 EMCF 的时间步和空间步是它的两次退化、相位闭合校正怎样把结果拉回一致的影像相位，
以及 moraine 怎样实现每一步。读者需要了解干涉图、相位包裹和最小费用流（MCF）解缠；怎样运行见
[05 相位解缠](../workflows/05_unwrap.md)。

## 记号

**下标表示空间。** $p, q$ 是点；$pq$ 是点的 Delaunay 三角网中由 $p$ 指向 $q$ 的边；$\tau$ 是三角网中的三角形，
$\partial\tau$ 是它的三条边；$\sigma_\tau(pq) = +1$ 表示边 $pq$ 的方向与沿 $\tau$ 逆时针绕行的方向一致，否则为 $-1$。

**上标表示时间。** $i, j, l$ 是影像；$ij$ 是像对，即干涉图"影像 $i$ − 影像 $j$"；$\theta$ 是像对网络中的回路，
例如 $(ij), (jl), (il)$；$\partial\theta$ 是回路中的像对，$\sigma_\theta(ij) = \pm 1$ 是像对在回路中的方向（这个例子里
$(ij)$、$(jl)$ 为 $+1$，$(il)$ 为 $-1$）。

**做差规则。** 对定义在（点, 影像）上的量 $X_p^i$，下标写两个点表示沿边做差，上标写两景表示两景相减：

$$
X_{pq}^i = X_q^i - X_p^i, \qquad
X_p^{ij} = X_p^i - X_p^j, \qquad
X_{pq}^{ij} = X_q^i - X_p^i - X_q^j + X_p^j .
$$

$X_{pq}^{ij}$ 是干涉图 $ij$ 沿边 $pq$ 的梯度，下文称为**混合差**。只定义在边上的量（例如代价 $c_{pq}$）也用下标 $pq$，
只定义在像对上的量（例如代价 $c^{ij}$）也用上标 $ij$。$\mathcal{W}(x) = x - 2\pi \operatorname{round}(x / 2\pi)$
把相位包裹到 $(-\pi, \pi]$。

**数据与未知数。** $\psi_p^i \in (-\pi, \pi]$ 是相位连接后第 $i$ 景在点 $p$ 的包裹相位，$N_p^i \in \mathbb{Z}$
是要求的整周，解缠相位是

$$
\Phi_p^i = \psi_p^i + 2\pi N_p^i .
$$

按做差规则，$\psi_{pq}^{ij}$ 是直接由包裹相位算出的混合差（不再包裹），观测到的干涉图梯度是
$\mathcal{W}(\psi_{pq}^{ij})$。包裹时丢掉的整周

$$
m_{pq}^{ij} = \frac{\psi_{pq}^{ij} - \mathcal{W}(\psi_{pq}^{ij})}{2\pi} \in \{-2, \dots, 2\}
$$

是已知的整数。解缠后，干涉图 $ij$ 沿边 $pq$ 的梯度相对观测值改动的整周数是

$$
k_{pq}^{ij} = \frac{\Phi_{pq}^{ij} - \mathcal{W}(\psi_{pq}^{ij})}{2\pi} = m_{pq}^{ij} + N_{pq}^{ij} .
$$

## 单幅干涉图的 MCF

单独解缠一幅干涉图 $ij$ 时，未知数是这幅干涉图在每个点的整周 $K_p^{ij} \in \mathbb{Z}$（它本应等于 $N_p^{ij}$），
每条边上的改动是 $m_{pq}^{ij} + K_{pq}^{ij}$，要求

$$
\min_{K^{ij}} \sum_{pq} c_{pq} \left| m_{pq}^{ij} + K_{pq}^{ij} \right| ,
$$

$c_{pq} > 0$ 是在边 $pq$ 上改一周的代价，和式取遍三角网的边。每一项只含两个未知数之差 $K_q^{ij} - K_p^{ij}$，
这类问题称为图上的 L1 势问题。

同一个问题也可以把改动 $k_{pq}^{ij}$ 当作未知数来写。改动后的梯度绕每个三角形必须闭合；绕三角形一圈
$\psi_{pq}^{ij}$ 的和为 0，所以闭合条件是

$$
\sum_{pq \in \partial\tau} \sigma_\tau(pq)\, k_{pq}^{ij} = -r_\tau^{ij}, \qquad
r_\tau^{ij} = \frac{1}{2\pi} \sum_{pq \in \partial\tau} \sigma_\tau(pq)\, \mathcal{W}(\psi_{pq}^{ij}) ,
$$

$r_\tau^{ij}$ 是干涉图 $ij$ 在三角形 $\tau$ 上的残差点。把每个三角形看成一个节点（凸包外再加一个地球节点），
$k_{pq}^{ij}$ 就是穿过边 $pq$、在两侧三角形之间的流，闭合条件是以 $r_\tau^{ij}$ 为供给的流量守恒，问题就是最小费用流
（Costantini 1998；稀疏点见 Costantini & Rosen 1999）。在单连通的三角网上两种写法等价：所有三角形都闭合，当且仅当
$k^{ij} - m^{ij}$ 是某个整数场的差。

隐含的假设是 Itoh 条件：绝大多数边上真实梯度的绝对值小于 $\pi$，即 $k_{pq}^{ij} = 0$，只在少数代价低的边上跳周。
每幅干涉图各解各的，像对网络的冗余用不上。

## 把解缠看成空间 × 时间的问题

### 数据块

点分布在二维平面上，影像排在时间轴上，格点 $(p, i)$ 构成一个"二维空间 × 一维时间"的数据块。三维解缠就是给块里的
每个格点定一个整数 $N_p^i$。

### 什么量是小的

块里有三种差：

- **影像的空间梯度** $\Phi_{pq}^i$（同一景、相邻两点）。$\psi^i$ 相对参考影像，离参考影像远的影像累积了大的形变，
  梯度不一定小。
- **一个点上的干涉相位** $\Phi_p^{ij}$（同一点、两景）。整幅干涉图可以跨很多周，还带一个任意常数，不一定小。
- **混合差** $\Phi_{pq}^{ij}$（两点 × 两景组成的小方块）。它是短像对沿短边的梯度，小。

```text
         点 p           点 q
景 i   (p,i) ──────── (q,i)
         │               │
景 j   (p,j) ──────── (q,j)
```

另一个角度是规范自由度。把 $N_p^i$ 换成 $N_p^i + M^i + L_p$（$M^i \in \mathbb{Z}$ 是第 $i$ 景在所有点上加的同一个
整数，$L_p \in \mathbb{Z}$ 是点 $p$ 在所有影像上加的同一个整数），所有干涉图的梯度都不变，数据区分不了这些解；
它们对应"没有参考点"和"没有参考影像"。这时 $N_{pq}^i$ 变化 $L_q - L_p$，$N_p^{ij}$ 变化 $M^i - M^j$，只有
$N_{pq}^{ij}$ 不变。随规范变化的代价项会让目标函数取决于一个任意的选择，所以**代价只能放在混合差上**。

把代价放在 $\Phi_{pq}^i$ 和 $\Phi_p^{ij}$ 这样的一维差上（数据立方体的经典三维解缠），每一项仍是两个未知数之差，
仍可以用网络流精确求解；但它假设的"这些差小"在 InSAR 中不成立。

### 理想的三维解缠

$$
\min_{N} \sum_{pq} \sum_{ij} c_{pq}^{ij} \left| m_{pq}^{ij} + N_{pq}^{ij} \right|
$$

$c_{pq}^{ij} > 0$ 是在混合差 $(pq, ij)$ 上改一周的代价；理想情况下，它随这条边、这个像对上真实梯度超出
$(-\pi, \pi]$ 的可能性减小（长边、低质量的点、时间间隔长的像对上便宜）。和式取遍三角网的边和网络中的像对。

解出的是每景的整周 $N_p^i$：所有干涉图由同一组 $N$ 相减得到，任何回路都自动闭合，直接得到每景的解缠相位；
规范自由度由参考点和参考影像固定。

### 残差形式

把改动 $k_{pq}^{ij}$ 当作未知数时，约束有两族。

**空间闭合**：对每个像对 $ij$ 和每个三角形 $\tau$，
$\sum_{pq \in \partial\tau} \sigma_\tau(pq)\, k_{pq}^{ij} = -r_\tau^{ij}$，与单幅 MCF 相同。

**时间闭合**：对每条边 $pq$ 和每个回路 $\theta$，

$$
\sum_{ij \in \partial\theta} \sigma_\theta(ij)\, k_{pq}^{ij} = -r_{pq}^\theta, \qquad
r_{pq}^\theta = \frac{1}{2\pi} \sum_{ij \in \partial\theta} \sigma_\theta(ij)\, \mathcal{W}(\psi_{pq}^{ij}) ,
$$

$r_{pq}^\theta$ 是边 $pq$ 上回路 $\theta$ 的时间残差。

若点的三角网单连通、$\theta$ 取遍像对网络的回路，那么 $k$ 同时满足两族约束，当且仅当存在整数场 $N$ 使
$k = m + N_{pq}^{ij}$。证明：由空间闭合，$k^{ij} - m^{ij}$ 是某个整数场 $K^{ij}$ 的空间差，取 $K^{ij}$ 在参考点为 0；
每条边上 $k - m$ 都满足时间闭合，沿空间路径累加，每个点的 $K_p^{ij}$ 也满足，所以 $K_p^{ij} = N_p^{ij}$。
残差形式与上面的势形式是同一个问题。

两类残差不是独立的。对每个三角形 $\tau$ 和回路 $\theta$ 交换求和的次序，得到

$$
\sum_{ij \in \partial\theta} \sigma_\theta(ij)\, r_\tau^{ij} = \sum_{pq \in \partial\tau} \sigma_\tau(pq)\, r_{pq}^\theta .
$$

一幅干涉图的残差点不会凭空消失：它要么接到同一回路中另一幅干涉图的同一个三角形，要么变成这个三角形某条边上的
时间残差。

### 几何图像

设像对网络是平面三角网（例如原文在时间—垂直基线平面上的 Delaunay 网络）。$S^*$ 是点三角网的对偶图：节点是三角形
$\tau$ 加地球节点，每条边 $pq$ 对应一条对偶边。$T^*$ 是像对三角网的对偶图：节点是回路 $\theta$ 加外部面，每个像对
$ij$ 对应一条对偶边。在乘积 $S^* \times T^*$ 中：

| $S^* \times T^*$ 中的元素 | 对应的量 |
|---|---|
| 顶点 $(\tau, \theta)$ | 残差守恒式 |
| 边 $(\tau, ij)$ 和 $(pq, \theta)$ | 残差 $r_\tau^{ij}$ 和 $r_{pq}^\theta$ |
| 方格 $(pq, ij)$ | 改动 $k_{pq}^{ij}$ |

残差是闭合的线（守恒式就是每个顶点流入等于流出），两族约束说的是"曲面 $k$ 的边界等于残差线"。**理想三维解缠就是求
以残差线为边界、代价最小的曲面。** 二维时残差是点、割是路径，最小费用流求的是代价最小的路径；到了三维，
"点 + 路径"变成"闭合线 + 曲面"。

### 为什么不是网络流

二维时每一项只含两个未知数之差。三维时每一项 $m_{pq}^{ij} + N_{pq}^{ij}$ 含四个未知数
$N_q^i, N_p^i, N_q^j, N_p^j$；在残差形式中，每个 $k_{pq}^{ij}$ 出现在两个空间约束（边 $pq$ 两侧的三角形）和若干个
时间约束（含 $ij$ 的回路）里。约束矩阵不是网络矩阵，问题不能化成最小费用流，是一般的整数规划。只有几个点、几景的
小例子里，映射 $N \mapsto N_{pq}^{ij}$ 的矩阵就已经不是全单模的，线性规划松弛会有半整数的最优解，所以也不能靠线性
规划精确求解。对上千万个点的数据，直接求解理想问题不现实。

## EMCF：两次退化

$N_{pq}^{ij}$ 的四个未知数可以按两种方式写成"差的差"。把其中一层的差当作独立的未知数，问题就又变回两个未知数之差，
可以用 MCF 求解。

记 $\mathcal{F}_S$ 为满足空间闭合的 $k$ 的集合，$\mathcal{F}_T$ 为满足时间闭合的集合，
$J(k) = \sum_{pq} \sum_{ij} c_{pq}^{ij} \left| k_{pq}^{ij} \right|$。理想问题是
$\min_{\mathcal{F}_S \cap \mathcal{F}_T} J$；去掉任何一族约束都是松弛：

$$
\min_{\mathcal{F}_S} J \le \min_{\mathcal{F}_S \cap \mathcal{F}_T} J, \qquad
\min_{\mathcal{F}_T} J \le \min_{\mathcal{F}_S \cap \mathcal{F}_T} J .
$$

### 逐幅解缠：去掉时间闭合

把 $N_p^{ij}$ 换成每个像对各自独立的整数场 $K_p^{ij}$，问题按像对解耦，就是上面的单幅 MCF（`mcf-pc`）。像对网络
没有回路时没有时间闭合约束，逐幅解缠与理想问题等价；这也是没有冗余就发现不了解缠错误的原因。

### 时间步：去掉空间闭合

按"先空间、后时间"分组：

$$
k_{pq}^{ij} = m_{pq}^{ij} + \underbrace{\left( N_q^i - N_p^i \right)}_{n_{pq}^i}
- \underbrace{\left( N_q^j - N_p^j \right)}_{n_{pq}^j} .
$$

$n_{pq}^i$ 是第 $i$ 景沿边 $pq$ 的整周梯度。把它当作每条边独立的未知数（按做差规则 $n_{pq}^{ij} = n_{pq}^i - n_{pq}^j$），
问题按边解耦，每条边 $pq$ 解一次：

$$
\min_{n_{pq}} \sum_{ij} c^{ij} \left| m_{pq}^{ij} + n_{pq}^{ij} \right| ,
$$

$c^{ij} > 0$ 是改动像对 $ij$ 一周的代价。这是像对图上的势问题：节点是影像，边是像对。丢掉的约束是
$n_{pq}^i = N_q^i - N_p^i$，即绕一个空间三角形一圈 $n_{pq}^i$ 的和为 0。点的网络是一棵树（没有三角形）时，时间步与
理想问题等价。

例：某条边上 $(ij)$、$(jl)$、$(il)$ 的真实梯度为 $0.6\pi$、$0.6\pi$、$1.2\pi$。包裹后 $(il)$ 变成 $-0.8\pi$，时间残差
$r_{pq}^\theta = (0.6\pi + 0.6\pi + 0.8\pi) / 2\pi = 1$。给 $(il)$ 加一周、给 $(ij)$ 或 $(jl)$ 减一周都能闭合，单位代价下
三种改法一样好；改哪一幅要靠其他回路（每景与后三景组成像对时，每个像对属于好几个三角回路）或代价（长像对更容易
混叠，改它更便宜）来定。短像对梯度小、几乎不混叠，时间步用它们确定长像对在每条边上的整周，所以长像对上梯度超过
$\pi$ 的边也能得到正确的值。

moraine 的时间步（`moraine/api/unwrap/emcf.py` 的 `_emcf_temporal`）从包裹后的影像梯度 $\mathcal{W}(\psi_{pq}^i)$
出发写这个问题，未知数与 $n_{pq}^i$ 相差一个已知整数（包裹 $\psi_{pq}^i$ 时丢掉的整周），是同一个问题。求解
（`moraine/api/unwrap/closure.py` 的 `_l1_fit`）先沿像对图的生成森林积分出每景的整数势，使树上像对的改动为 0；
所有改动都为 0 时（这条边上的回路已经全部闭合，大多数边如此）直接结束。否则解对偶问题，即像对图上容量为 $c^{ij}$
的最小费用循环流：用 Bellman–Ford 找负圈并消去，最后的最短路距离给出最优势。像对网络恰好是平面三角网时，它与原文在
三角网对偶图上的 MCF 是同一个问题；所以 moraine 不需要像对三角网，任何像对网络都能用（决策 0020）。

代价由 `temporal_cost` 选择：`constant` 时 $c^{ij} = 1$；`length` 时

$$
c^{ij} = \operatorname{clip}\left( \operatorname{round}\left( 4\, \frac{\overline{\Delta t}}{\Delta t^{ij}} \right),\ 1,\ 20 \right),
$$

其中 $\Delta t^{ij} = |t^i - t^j|$ 是像对 $ij$ 的时间间隔，$t^i$ 是第 $i$ 景的获取时间，$\overline{\Delta t}$ 是所有像对
时间间隔的中位数。短像对贵，长像对优先被改。

### 空间步：以时间步为参照

按"先时间、后空间"分组：

$$
k_{pq}^{ij} = m_{pq}^{ij} + \underbrace{\left( N_q^i - N_q^j \right)}_{K_q^{ij}}
- \underbrace{\left( N_p^i - N_p^j \right)}_{K_p^{ij}} .
$$

把 $K_p^{ij}$ 当作每个像对独立的未知数，就是逐幅解缠。EMCF 的空间步（`_emcf_spatial`）不以包裹值为参照，而是让每幅
干涉图的整周梯度尽量接近时间步的结果 $\hat n_{pq}^i$，每个像对 $ij$ 解一次：

$$
\min_{K^{ij}} \sum_{pq} c_{pq} \left| K_{pq}^{ij} - \hat n_{pq}^{ij} \right| .
$$

实现上，时间步修正后的梯度 $\mathcal{W}(\psi_{pq}^{ij}) + 2\pi \left( m_{pq}^{ij} + \hat n_{pq}^{ij} \right)$ 绕三角形的
残差点是 $\hat r_\tau^{ij} = \sum_{pq \in \partial\tau} \sigma_\tau(pq)\, \hat n_{pq}^{ij}$，由最小费用流（逐次最短路，
`moraine/api/unwrap/mcf.py`）消去，再沿三角形从第一个点 $p_0$ 积分；$p_0$ 的值取 $\psi_{p_0}^i - \psi_{p_0}^j$（不包裹）。

代价由 `spatial_cost` 选择：$c_{pq} = 1 + \operatorname{round}(99\, \eta_{pq})$，$\eta_{pq} \in [0, 1]$ 是边的可靠度。
`length` 时 $\eta_{pq} = \min(1, \bar L / L_{pq})$，$L_{pq}$ 是边长，$\bar L$ 是边长的中位数；`weight` 时
$\eta_{pq} = \min(w_p, w_q)$，$w_p \in [0, 1]$ 是点的权重（例如时间相干性）；`length+weight` 时取两者之积；
`constant` 时 $c_{pq} = 1$。可靠的边贵，跳周优先放在长边和低质量的点上；通往地球节点的边再乘 `earth_cost`。

若空间步不加任何改动（所有边上 $K_{pq}^{ij} = \hat n_{pq}^{ij}$），沿同一棵积分树累加得到
$K_p^{ij} = A_p^i - A_p^j$，$A_p^i$ 是从 $p_0$ 沿积分路径把 $\hat n_{pq}^i$ 累加的结果，所有回路在每个点上自动闭合。
回路不闭合只可能来自空间步。

### 闭合校正：回到一致的影像相位

时间步的结果在 $\mathcal{F}_T$ 中，空间步把它投影到 $\mathcal{F}_S$。空间步加的改动
$f_{pq}^{ij} = K_{pq}^{ij} - \hat n_{pq}^{ij}$ 只有能写成 $F_{pq}^i - F_{pq}^j$（$F_{pq}^i$ 是第 $i$ 景自己的整数改动）时，
回路才仍然闭合；逐幅求解不保证这一点。在上面的几何图像里，时间步沿 $T^*$ 逐条确定曲面（每条边 $pq$ 一条），
空间步沿 $S^*$ 逐条补方格（每个像对一条）；补上的方格 $(pq, ij)$ 的边界也落在时间边 $(pq, \theta)$ 上，同一回路里的
几幅干涉图补得不一样，时间闭合就被破坏。

相位闭合校正（`unwrap-correct-closure-pc`，决策 0021、0022）把结果拉回 $\mathcal{F}_S \cap \mathcal{F}_T$。它先在
每个点上求

$$
\min_{N_p} \sum_{ij} \left| N_p^{ij} - K_p^{ij} \right| ,
$$

得到每幅干涉图的改动 $\mu_p^{ij} = N_p^{ij} - K_p^{ij}$。然后在每个区域 $R$（去掉长于 $4 \bar L$ 的边后的连通分量）内，
对每个像对取 $\mu_p^{ij}$ 的中位数 $\bar\mu_R^{ij}$ 整体平移，因为空间解缠的错误是整块平移的；少于 30 个点的区域逐点
用自己的 $\mu_p^{ij}$。最后逐点再解一次，得到每景的整周 $N_p^i$ 和相对参考影像 $i_0$ 的相位
$\Phi_p^{i i_0} = \Phi_p^i - \Phi_p^{i_0}$。结果满足两族约束，但不一定是理想问题的最优解。

### 对照

| 方法 | 未知数 | 每一项 | 能否用 MCF | 满足的约束 |
|---|---|---|---|---|
| 理想三维 | $N_p^i$ | 四个未知数 | 不能 | 两族 |
| 逐幅解缠（`mcf-pc`） | $K_p^{ij}$，每个像对独立 | 两个未知数之差 | 能 | 空间闭合 |
| 时间步 | $n_{pq}^i$，每条边独立 | 两个未知数之差 | 能 | 时间闭合 |
| 空间步 | $K_p^{ij}$，以时间步为参照 | 两个未知数之差 | 能 | 空间闭合 |
| 闭合校正 | $N_p^i$，每个点独立 | 两个未知数之差 | 能 | 两族，不一定最优 |

## 例子：一景自己的残差点

设第 $i$ 景在三角形 $\tau$ 中有噪声，只有这一景在这里有残差点，其他影像都光滑。含 $i$ 的每个像对都有
$r_\tau^{ij} = \pm 1$，各边的时间残差都为 0。在 $T^*$ 中，与 $i$ 相连的像对的对偶边围成一圈，所以残差线是位于
$\tau$、在 $T^*$ 中绕影像 $i$ 一圈的闭合线。

- **理想解**：最小曲面是"$S^*$ 中从 $\tau$ 出发的一条路径 $\gamma$" × "这一圈"，即所有含 $i$ 的干涉图在同一条路径
  $\gamma$ 上跳周，回路照样闭合。
- **时间步**：含 $i$ 的像对之间没有矛盾，时间步不改动任何干涉图的梯度。
- **空间步**：每幅 $(ij)$ 在 $\tau$ 上都有残差点，各自选一条割线 $\gamma_j$。只要 $\gamma_j \neq \gamma_l$，两条割线之间
  的点上回路 $(ij) + (jl) - (il)$ 就差一周，留给闭合校正。

## 假设与局限

- 时间步假设每条边上按代价加权的多数像对满足 $\left| \Phi_{pq}^{ij} \right| < \pi$，即时间维的 Itoh 条件。多数像对都混叠时
  （跨空缺的长边、陡峭的形变梯度、某一景很强的大气），L1 会跟着多数改错。
- 形如 $n_{pq}^i - n_{pq}^j$ 的改动不破坏任何回路。若第 $i$ 景在某条边上与其他所有景的真实梯度差都比包裹值多一周，
  时间步取 $n = 0$ 的代价为 0，看不到这个错误。错误只在一条边上时，它在含 $i$ 的干涉图里留下残差点，由空间步处理；
  错误在空间上也自洽时（例如第 $i$ 景在一整块区域上差一周），它是没有边界的曲面，加上它只会增加代价，理想问题也不会
  纠正它。工作流指南里的合成数据上，没有闭合差的点中仍有 0.4–9 % 是错的，所以要结合时间相干性等点的质量来用结果。
- 短像对约束长像对：网络里要有足够多梯度小的短像对。

## 与原始 EMCF 的关系

- 原文（Pepe & Lanari 2006）对影像在时间—垂直基线平面上做 Delaunay 三角化得到像对网络，时间步是这个三角网对偶图上
  的 MCF，每条空间边解一次；空间步以时间步的结果为起点，在方位—距离平面上逐幅做 MCF。moraine 用任意像对网络（不用
  垂直基线，示例里每景与后三景组成像对），把时间步写成势问题；平面三角网上两者的最优值相同（决策 0020）。
- 原文的输入是多视干涉图。设 $\varphi_p^{ij}$ 是多视干涉图 $ij$ 在点 $p$ 的包裹相位，闭合相位
  $\mathcal{W}(\varphi_p^{ij} + \varphi_p^{jl} - \varphi_p^{il})$ 一般不为 0，时间残差里混有噪声。moraine 的干涉图由相位
  连接后的影像相位构成，闭合相位严格为 0（模 $2\pi$），时间残差
  $r_{pq}^\theta = -\sum_{ij \in \partial\theta} \sigma_\theta(ij)\, m_{pq}^{ij}$ 只来自梯度的包裹，$m_{pq}^{ij}$ 可以直接由
  影像相位写出。
- 两步之后回路仍不闭合的地方，moraine 用单独的相位闭合校正处理（决策 0021），它也能用在逐幅解缠的结果上。

## 在 moraine 中

- 命令：`image-pairs` 生成像对网络，`emcf-pc` 做时间步和空间步，`unwrap-correct-closure-pc` 做闭合校正并输出每景的
  相位；`mcf-pc` 是逐幅解缠，`gamma-mcf-pt` 用 GAMMA 的 `mcf_pt`。参数见[命令行参考](../cli/pu.md)，运行步骤和预期结果
  见 [05 相位解缠](../workflows/05_unwrap.md)。
- 代码：`moraine/api/unwrap/emcf.py`（两步）、`moraine/api/unwrap/closure.py`（像对图上的 L1 势问题和闭合校正）、
  `moraine/api/unwrap/mcf.py`（点三角网上的最小费用流）。
- 决策：[0012](../decisions/0012-network-emcf-inversion.md)（冗余网络与 EMCF）、
  [0013](../decisions/0013-own-delaunay-and-mcf.md)（Delaunay 三角网与逐次最短路 MCF）、
  [0020](../decisions/0020-emcf-any-image-pair-network.md)（任意像对网络、两步的单元、代价）、
  [0021](../decisions/0021-closure-correction-by-regions.md)（按区域的闭合校正）、
  [0022](../decisions/0022-closure-correction-outputs-phase-time-series.md)（输出每景的相位）。

## 参考文献

- M. Costantini, "A novel phase unwrapping method based on network programming", IEEE Transactions on Geoscience and
  Remote Sensing 36(3), 1998.
- M. Costantini, P. A. Rosen, "A generalized phase unwrapping approach for sparse data", IGARSS 1999.
- A. Pepe, R. Lanari, "On the extension of the minimum cost flow algorithm for phase unwrapping of multitemporal
  differential SAR interferograms", IEEE Transactions on Geoscience and Remote Sensing 44(9), 2006.
- Z. Yunjun, H. Fattahi, F. Amelung, "Small baseline InSAR time series analysis: unwrapping error correction and noise
  reduction", Computers & Geosciences 133, 104331, 2019.
