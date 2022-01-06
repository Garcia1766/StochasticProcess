# 随机过程 Project

> 李阳崑  2021310749

[TOC]

## 1

### (1)

定义第 $i$ 个时刻二者距离的变化为随机变量 $Y_i$ , 则 $P\{Y_i=-2\} = p_1(1-p_2)$ , $P\{Y_i=2\} = (1-p_1)p_2$ , $P\{Y_i=0\} = p_1p_2+(1-p_1)(1-p_2)$ . 即距离变化的取值只可能是 $\{-2, 0, 2\}$ , 因此可能碰撞的条件是 $a_2-a_1$ 为偶数。

### (2)

#### $E[T_c]$

$P\{Y_i=-2\} = p^2$ , $P\{Y_i=2\} = (1-p)^2$ , $P\{Y_i=0\} = 2p(1-p)$ . $E[Y_i]=2(1-2p)$ . $Y_0=0$ .

令 $S_n=\sum\limits_{i=0}^n Y_i$ , 则 $E[S_n] = 2n(1-2p)$ . 停时 $T_c=\min\{n:S_n=a_1-a_2\}$ .

令 $X_n = \sum\limits_{i=0}^n (Y_i-E[Y_i]) = S_n-2n(1-2p) = S_n-E[S_n]$ , $\{X_n\}$ 关于 $\{Y_n\}$ 是鞅.

由于 $E[|X_{n+1}-X_n|\ |\mathcal{F}_{n}] = E[|Y_{n+1}-2(1-2p)|\ |\mathcal{F}_{n}] < 2 - 2(1-2p)$ 有界, 满足停时定理3的条件，故

$0=E[X_0] = E[X_{T_c}] = E[S_{T_c}-2(1-2p)T_c] = a_1-a_2-2(1-2p)E[T_c]$ $\Rightarrow$ $E[T_c] = \dfrac{a_2-a_1}{2(2p-1)}$ .

#### $var[T_c]$

令 $Z_n=\sum\limits_{k=0}^n\Big(X_k^2-E[X_k^2|\mathcal{F}_{k-1}]\Big)$ . 化简 $E[X_k^2|\mathcal{F}_{k-1}]$ : 
$$
\begin{aligned}
X_k^2 &= (X_{k-1}+Y_k-2(1-2p))^2 \\
&= X_{k-1}^2+Y_k^2+4(1-2p)^2 + 2X_{k-1}Y_k - 4(1-2p)X_{k-1} - 4(1-2p)Y_k \\
E[X_k^2|\mathcal{F}_{k-1}] &= X_{k-1}^2+E[Y_k^2] + 4(1-2p)^2 + 2X_{k-1}E[Y_k] - 4(1-2p)X_{k-1} - 4(1-2p)E[Y_k] \\
&= X_{k-1}^2+4(p^2+(1-p)^2) + 4(1-2p)^2 + 4(1-2p)X_{k-1} - 4(1-2p)X_{k-1} - 8(1-2p)^2 \\
&= X_{k-1}^2 + 8p(1-p)
\end{aligned}
$$
故 $Z_n=\sum\limits_{k=0}^n \Big(X_k^2 - X_{k-1}^2 - 8p(1-p) \Big) = X_n^2 - 8np(1-p)$ . 由课本P47鞅的构造方法可知 $Z_n$ 关于 $Y_n$ 是鞅. 且可验证 $E[|Z_{n+1}-Z_n|\ |\mathcal{F}_{n}]$ 有界，停时定理3的条件满足。因此有 $E[Z_{T_c}] = E[Z_0]$ .

$E[Z_0] = E[X_0^2] = 0$ , $E[Z_{T_c}] = E[X_{T_c}^2 - 8p(1-p)T_c] = E[X_{T_c}^2]-8p(1-p)E[T_c]$ . $E[X_{T_c}^2] = E[(S_{T_c}-2T_c (1-2p))^2] = E[(a_1-a_2-2(1-2p)T_c)^2]$ . 

$\Rightarrow$ $E[(a_1-a_2-2(1-2p)T_c)^2] - 8p(1-p)E[T_c] = 0$ $\Rightarrow$ $4p^2(1-p)^2E[T_c^2] - 4p(1-p)(a_1-a_2)E[T_c] + (a_1-a_2)^2 = 8p(1-p)E[T_c]$

$\Rightarrow$ $E[T_c^2]=\dfrac{(a_2-a_1)^2}{4(2p-1)^2} + \dfrac{p(1-p)(a_2-a_1)}{(2p-1)^3}$ .

$\Rightarrow$ $var[T_c] = E[T_c^2]-E^2[T_c] = \dfrac{p(1-p)(a_2-a_1)}{(2p-1)^3}$ .

### 实验仿真 (2)

首先初步看一下 $E[T_c]$ , $var[T_c]$ 的收敛速度：

<img src="figs/figs_1-2-0/1_0.6_1000_1217_ET.png" style="zoom:9%;" /><img src="figs/figs_1-2-0/1_0.6_1000_1217_varT.png" style="zoom:9%;" />

<img src="figs/figs_1-2-0/1_0.7_1000_1217_ET.png" style="zoom:9%;" /><img src="figs/figs_1-2-0/1_0.7_1000_1217_varT.png" style="zoom:9%;" />

<img src="figs/figs_1-2-0/1_0.8_1000_1217_ET.png" style="zoom:9%;" /><img src="figs/figs_1-2-0/1_0.8_1000_1217_varT.png" style="zoom:9%;" />

<img src="figs/figs_1-2-0/1_0.9_1000_1217_ET.png" style="zoom:9%;" /><img src="figs/figs_1-2-0/1_0.9_1000_1217_varT.png" style="zoom:9%;" />

可以看到，在 $b$ 固定的条件下，$p$ 越小（越接近 $0.5$）收敛越慢，且 $var[T_c]$ 比 $E[T_c]$ 收敛慢。

因此取模拟次数为 $10000$ 次，固定 $b=1$ 画 $E[T_c]$ 和 $p$ 的关系曲线：

<img src="figs/figs_1-2/1_10000_1217_ET.png" style="zoom:9%;" /><img src="figs/figs_1-2/1_10000_1217_varT.png" style="zoom:9%;" />

其中 $p$ 取 $[0.5, 0.995]$ 均匀地取 $100$ 个值，红线是仿真结果，蓝线是理论结果 $E[T_c] = \dfrac{1}{2p-1}$ , $var[T_c] = \dfrac{2p(1-p)}{(2p-1)^3}$ . 可看到基本一致。

固定 $p=0.75$ 画 $E[T_c]$ 和 $b$ 的关系曲线：

<img src="figs/figs_1-2/0.75_10000_1217_ET.png" style="zoom:9%;" /><img src="figs/figs_1-2/0.75_10000_1217_varT.png" style="zoom:9%;" />

其中 $b$ 取 $[1, 100]$ 这 $100$ 个整数值。红线是仿真结果，蓝线是理论结果 $E[T_c]=2b$ , $var[T_c] = 3b$ . 可看到基本一致。

### (3)

#### 证 $E[T_c]$ 不存在

$P\{Y_i=-2\} = \dfrac{1}{4}$ , $P\{Y_i=2\} = \dfrac{1}{4}$ , $P\{Y_i=0\} = \dfrac{1}{2}$ . $E[Y_i]=0$ . $Y_0=0$ .

令 $S_n=\sum\limits_{i=0}^n Y_i$ , 则 $E[S_n] = 0$ . $S_n$ 关于 $Y_n$ 是鞅. 停时 $T_c=\min\{n:S_n=a_1-a_2\}$ .

$E[|S_{n+1}-S_n|\ |\mathcal{F}_n] = E[|Y_{n+1}|]=1$ 有界，因此若 $E[T_c] < \infin$ , 停时定理3的条件满足，就有 $E[S_{T_c}] = E[S_0] = 0$ . 但由定义, $E[S_{T_c}] = a_1-a_2$ , 矛盾，因此 $E[T_c]=\infin$ , 即 $E[T_c]$ 不存在.

#### 求 $T_c$ 概率分布

把问题的数值表示简化一下，等价于以下随机游走问题：

$b=\dfrac{a_2-a_1}{2} \in \mathbb{N}$ , 一个粒子从原点出发，每时刻移动的距离 $Y_i$ 对应的概率为：$P\{Y_i=-1\} = \dfrac{1}{4}$ , $P\{Y_i=1\} = \dfrac{1}{4}$ , $P\{Y_i=0\} = \dfrac{1}{2}$ . 粒子首次撞到 $b$ 的时刻记为 $T$ , 求 $T$ 的概率分布。

为解决以上问题，记粒子首次撞到 $v$ 经过的时间为 $T_v$ ，则有递推关系：
$$
P(T_v=n) = \dfrac{1}{4}P(T_{v-1}=n-1) + \dfrac{1}{4}P(T_{v+1}=n-1) + \dfrac{1}{2}P(T_{v}=n-1)
$$
这是因为，考虑经过1个时刻，若粒子向右移动1，则等价于目标点移至 $v-1$ , 接下来还需经过 $n-1$ 个时刻恰好首次撞到目标点，概率为 $P(T_{v-1}=n-1)$ ；若粒子向左移动1，则等价于目标点移至 $v+1$ , 接下来还需经过 $n-1$ 个时刻恰好首次撞到目标点，概率为 $P(T_{v+1}=n-1)$ ；若粒子不动，则目标点还是 $v$ , 接下来还需经过 $n-1$ 个时刻恰好首次撞到目标点，概率为 $P(T_{v}=n-1)$ 。

考虑序列 $\{P(T_v=n)\}$ 的 $z$ 变换 $f_v(z) = \sum\limits_{n=-\infin}^\infin P(T_v=n)z^{n}$ , 对以上递推关系的每项都做 $z$ 变换，得
$$
f_v(z) = \dfrac{1}{4}zf_{v-1}(z) + \dfrac{1}{4}zf_{v+1}(z) + \dfrac{1}{2}zf_v(z)
$$
考察 $f_v(z)$ 和 $f_{v-1}(z)$ 的关系，即考察 $P(T_v=n)$ 和 $P(T_{v-1}=n)$ 的关系：粒子首次撞到 $v$ 的耗时 $T_v$ , 等于粒子首次撞到 $1$ 的耗时 $T_1$ , 加上从 $1$ 出发首次撞到 $v$ 的耗时。而后者等于从原点出发首次撞到 $v-1$ 的耗时 $T_{v-1}$ , 即有 $T_v = T_1 + T_{v-1}$ . 因此有
$$
\begin{aligned}
P(T_v=n) &= \sum_{k=-\infin}^{\infin} P(T_1=k)P(T_{v-1}=n-k) = P(T_1=n) \ast P(T_{v-1}=n)
\end{aligned}
$$
两边进行 $z$ 变换 , 得 $f_v(z) = f_1(z)f_{v-1}(z)$ . 因此有 $f_1(z)f_{v-1}(z) = \dfrac{1}{4}zf_{v-1}(z) + \dfrac{1}{4}zf_1^2(z)f_{v-1}(z) + \dfrac{1}{2}zf_1(z)f_{v-1}(z)$ , 即
$$
f_1^2(z) - 2(\dfrac{2}{z}-1)f_1(z) + 1 = 0 \\
\Rightarrow\quad f_1(z) = \dfrac{2}{z}-1 \pm \sqrt{(\dfrac{2}{z}-1)^2 - 1}
$$
根据 $f_v(z)$ 的定义 $f_v(z) = \sum\limits_{n=-\infin}^\infin P(T_v=n)z^{n} = \sum\limits_{n=1}^\infin P(T_v=n)z^{n}$ , 因为 $\sum\limits_{n=1}^\infin P(T_v=n) = 1$ , 故当 $z<1$ 时必有 $f_v(z) < 1,\ \forall v\ge 1$ . 因此 $f_1(z) = \dfrac{2}{z}-1 - \sqrt{(\dfrac{2}{z}-1)^2 - 1}$ . 故得
$$
f_v(z) = f_1^v(z) = \left(\dfrac{2}{z}-1 - \sqrt{(\dfrac{2}{z}-1)^2 - 1}\right)^v
$$
$P(T_b = n)$ 就是上式关于 $z$ 做泰勒展开后 $z^{n}$ 项的系数。回到原问题，若 $a_2-a_1$ 为偶数，这样求得的概率分布就是 $T_c$ 的概率分布；若 $a_2-a_1$ 是奇数，则 $P(T_c=\infin) = 1$ .

### 实验仿真 (3)

原始概率分布求解较为麻烦，不过我们可以验证其 $z$ 变换是否和理论 $z$ 变换后的结果一致。具体来说，我们用仿真程序模拟 $10000$ 次游走，统计出 $P(T_c=n)$ 序列，计算 $\hat{f_v}(z) = \sum\limits_{n=1}^\infin P(T_v=n)z^{n}$ 是否和理论推导出的概率分布的 $z$ 变换 $f_v(z) = \left(\dfrac{2}{z}-1 - \sqrt{(\dfrac{2}{z}-1)^2 - 1}\right)^v$ 一致。对每个 $b$ 绘制二者的函数图像进行比较：

<img src="figs/figs_1-3/z_1_10000_1218.png" style="zoom:9%;" /><img src="figs/figs_1-3/z_2_10000_1218.png" style="zoom:9%;" /><img src="figs/figs_1-3/z_3_10000_1218.png" style="zoom:9%;" /><img src="figs/figs_1-3/z_4_10000_1218.png" style="zoom:9%;" /><img src="figs/figs_1-3/z_5_10000_1218.png" style="zoom:9%;" />

以上分别是 $z=\{1, 2, 3, 4, 5\}$ 的结果。红线是仿真得到的 $\hat{f_v}(z)$ ，蓝线是理论计算得到的 $f_v(z)$ . 可看到基本一致。

关于仿真程序中的截断步数：

随机游走可能出现游走步数极大的情形，如刚开始实验时我取 $b=1$ 就遇到了游走步数超过 $10^{9}$ 步的情况，一来使得游走部分的模拟耗时过长，二来使得 $z$ 变换 $\hat{f_v}(z) = \sum\limits_{n=1}^\infin P(T_v=n)z^{n}$ 的计算耗时过长。这种过长耗时的计算是无必要的，因为当 $n$ 过大时，$z^{n}$ 过小 ($z<1$) , 小于 python 的浮点数精度时就等于零，对结果无贡献。因此在仿真时，可采用游走步数超过某一定值就截断的方式，使此次游走不对 $z$ 变换的结果做出贡献。

截断步数 $N$ 的选取需要选择合适的值：若过大，则 $z^n$ 下溢，造成时间浪费；若过小，则当 $z$ 不太小时，$z^N$ 不太小，$z$ 变换中舍去未计算的部分 $\sum\limits_{n=N}^\infin P(T_v=n)z^{n}$ 使得结果有误差。我绘制了 $z\in[0.5, 0.995]$ 的函数图像，记游走 $x$ 步后 $z^x$ 下溢，则
$$
z^x = 2^{-308}\ \Rightarrow\ x=\dfrac{-92.72}{\lg z}
$$
$z=0.5$ 时，$x=308$ ; $z=0.995$ 时，$x=42592$ . 我折中选取 $N=5000$ , 即游走超过 $5000$ 步就截断，这样可以使 $z\le 0.955$ 时被截断的部分大约都是下溢了的，保证精确性的同时缩短程序运行时间。

在上面5张图中，随着 $b$ 增大，红线（仿真结果）逐渐由略高于蓝线（理论推导结果）变为略低于蓝线。这是因为当 $b$ 较小时，平均游走步数较低，截断掉的 $5000$ 步以上的部分几乎不对结果造成影响，系统误差使得红线高于蓝线；而当 $b$ 较大时，平均游走步数较大，截断掉的 $5000$ 步以上的部分使得由仿真结果计算得到的 $z$ 变换会损失掉一部分，因此红线的值相对变小了。

## 2

记 $X_n$ , $Y_n$ 分别是 $n$ 个时刻后两粒子的横、纵坐标差，令 $Z_n=X_n+Y_n$ , $W_n = X_n-Y_n$ , 则每个时刻 $Z_n$ 和 $W_n$ 的变化是随机变量，服从分布：
$$
P(\Delta Z_i = 2) = \dfrac{1}{4},\ P(\Delta Z_i = 0) = \dfrac{1}{2},\ P(\Delta Z_i = -2) = \dfrac{1}{4} \\
P(\Delta W_i = 2) = \dfrac{1}{4},\ P(\Delta W_i = 0) = \dfrac{1}{2},\ P(\Delta W_i = -2) = \dfrac{1}{4}
$$
令 $c=\dfrac{(a_2-a_1) + (b_2-b_1)}{2} \in \mathbb{N}$ , $d=\dfrac{(a_2-a_1) - (b_2-b_1)}{2} \in \mathbb{N}$ , 此问题等价于以下二维随机游走问题：

一个粒子从 $(0,0)$ 出发 , 每个时刻在 $X$ , $Y$ 方向上的位移是独立同分布随机变量 $X_i$ , $Y_i$ , 服从分布：
$$
P(X_i = 1) = \dfrac{1}{4},\ P(X_i = 0) = \dfrac{1}{2},\ P(X_i = -1) = \dfrac{1}{4} \\
P(Y_i = 1) = \dfrac{1}{4},\ P(Y_i = 0) = \dfrac{1}{2},\ P(Y_i = -1) = \dfrac{1}{4} \\
$$
粒子首次碰到 $(c,d)$ 坐标用时 $T$ , 求 $T$ 的概率分布。 

记粒子首次撞到 $(u,v)$ 经过的时间为 $T_{u, v}$ ，则有递推关系：
$$
\begin{aligned}
P(T_{u, v} = n) &= \dfrac{1}{16}P(T_{u-1, v-1} = n-1) + \dfrac{1}{8}P(T_{u-1, v} = n-1) + \dfrac{1}{16}P(T_{u-1, v+1} = n-1) \\
&+ \dfrac{1}{8}P(T_{u, v-1} = n-1) + \dfrac{1}{4}P(T_{u, v} = n-1) + \dfrac{1}{8}P(T_{u, v+1} = n-1) \\
&+ \dfrac{1}{16}P(T_{u+1, v-1} = n-1) + \dfrac{1}{8}P(T_{u+1, v} = n-1) + \dfrac{1}{16}P(T_{u+1, v+1} = n-1) 
\end{aligned}
$$
理由同第1题第3问。考虑序列 $\{P(T_{u, v} = n)\}$ 的 $z$ 变换 $f_{u, v}(z) = \sum\limits_{n=-\infin}^{\infin} P(T_{u, v} = n)z^{n}$ . 对上试每项做 $z$ 变换，得
$$
\begin{aligned}
f_{u, v}(z) &= \dfrac{1}{16}zf_{u-1, v-1}(z) + \dfrac{1}{8}zf_{u-1, v}(z) + \dfrac{1}{16}zf_{u-1, v+1}(z) \\
&+ \dfrac{1}{8}zf_{u, v-1}(z) + \dfrac{1}{4}zf_{u, v}(z) + \dfrac{1}{8}zf_{u, v+1}(z) \\
&+ \dfrac{1}{16}zf_{u+1, v-1}(z) + \dfrac{1}{8}zf_{u+1, v}(z) + \dfrac{1}{16}zf_{u+1, v+1}(z)
\end{aligned}
$$
采用类似第1题的思路，考察 $f_{u, v}(z)$ 和 $f_{u-1, v}(z)$ 和 $f_{u, v-1}(z)$ 的关系，但发现和1维情况不同的是：

在1维情况下，粒子在首次碰到 $v$ 点之前一定会碰到 $v-1$ 点，所以才有 $f_v(z) = f_1(z)f_{v-1}(z)$ 的关系。但在2维情况下，粒子在首次碰到 $(u, v)$ 点之前，不一定会碰到 $(u-1, v)$ 或 $(u, v-1)$ , 有可能只碰到过 $(u+1, v)$ 或 $(u, v+1)$ . 因此，无法仅用 $f_{u-1, v}(z)$ 和 $f_{u, v-1}(z)$ 表示 $f_{u, v}(z)$ , 后者还和 $f_{u+1, v}(z)$ 和 $f_{u, v+1}(z)$ 相关。因此也就无法求出以上递推关系的解析解。

## 3

不妨设 $a_1=0$, $a_2=a$ .

### (1)

两个粒子在时间 $t$ 内发生的位移差 $B_1(t) - B_2(t)$ 等价于一个粒子做布朗运动 $B(2t)$ .

因此 $X_2 - X_1 \sim B(2t)+a$ , 故有 $T_c = \min\limits_{t} \{t:B(2t)=a\}$ . 由定理5.6，可得 $T_c$ 的pdf为
$$
f_{T_c}(t) = \dfrac{a}{2\sqrt{\pi}} e^{-\frac{a^2}{4t}} t^{-\frac{3}{2}},\quad t>0
$$
拉普拉斯变换为 $e^{-a\sqrt{s}},\ s>0$ . 

对于 $X_c$ ，考虑二维布朗运动 $(Z_1(t), Z_2(t)) = \dfrac{1}{\sqrt{2}}(X_2(t)-X_1(t)+a, X_2(t)+X_1(t)-a)$ , 则 $T_c=\min\limits_{t}\{t:Z_1(t)=\dfrac{a}{\sqrt{2}}\}$ , $X_1(T_c)=X_2(T_c)=X_c$ . $Z_2(T_c) = \dfrac{X_2(T_c)+X
_1(T_c)-a}{\sqrt{2}} = \dfrac{2X_c-a}{\sqrt{2}}$ . 记 $\dfrac{a}{\sqrt{2}}=d$ , 则 $Z_2(T_c) = \sqrt{2}X_c - d$ 是 $d$ 的函数，记为 $Z(d)$ . 由课本5.12.2布朗运动的从属过程知 $Z(d)\sim Cauchy(d)$ , 故有
$$
P(X_c<x) = P(Z(d) < \sqrt{2}x-d) = \dfrac{1}{\pi}\arctan(\dfrac{2x-a}{a}) +\dfrac{1}{2}
$$
因此 $X_c$ 的pdf是
$$
f_{X_c}(x) = \dfrac{2a}{\pi((2x-a)^2 + a^2)}
$$


第1题中对 $f_v(z)$ 进行泰勒展开，得 $z^n$ 项的系数也就是 $P(T_b=n)$ 为
$$
P(T_b=n) = \binom{2n}{n-b} \dfrac{b}{n2^{2n}}
$$
本题中令 $a=1$ , 时间间隔取为 $\Delta t = \dfrac{1}{b^2}$ , 可知对应得离散初始距离为 $\dfrac{a}{\sqrt{\Delta t}}=b$ . 此时对应得概率分布为
$$
P(n\Delta t\le T_c < (n+1)\Delta t) = f_{T_c}(n\Delta t) \Delta t = \dfrac{b}{2\sqrt{\pi}}e^{-\frac{b^2}{4n}}n^{-\frac{3}{2}}
$$
$\Delta t \rightarrow 0$ 时 $b\rightarrow \infin$ , 两式相除并代入Stirling公式，得
$$
\dfrac{P(n\Delta t\le T_c < (n+1)\Delta t)}{P(T_b=n)} = \sqrt{\dfrac{4n^2-b^2}{4n^2}} \left(1-\dfrac{b^2}{4n^2}\right)^n \left(\dfrac{2n+b}{2n-b}\right)^b e^{-\frac{b^2}{4n}}
$$
当 $n\rightarrow \infin$ 时上式右端趋近于 $1$ . 即极限情况下一维离散随机游走得停时分布和一维布朗运动的停时分布一致。



## 代码说明

仿真代码用 python 完成。`results_*` 文件夹下是仿真所得的原始结果，`figs_*` 文件夹下是图像。
