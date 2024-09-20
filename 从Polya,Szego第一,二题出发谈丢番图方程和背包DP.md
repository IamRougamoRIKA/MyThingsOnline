# 引入
```ad-question
title:你能以多少方式用掉100美分？？你可以用的币种有1美分，5美分，10美分，25美分和50美分.(Polya,Szego第一题)
```
虽然是数学分析的题目
但是在计算机中是个常见的**背包dp示例**
```python
#背包dp做法,复杂度:O(nk)
def count_ways_dp(coins, total): 
# 初始化一个数组，dp[i]表示总金额为i的方法数 
dp = [0] * (total + 1) dp[0] = 1 # 总金额为0有一种方法，即不使用任何硬币
# 遍历每种硬币 
for coin in coins:
# 更新dp数组 
    for amount in range(coin, total + 1): 
        dp[amount] += dp[amount - coin] 
    return dp[total] 
# 调用函数计算方法数
 ways_dp = count_ways_dp([1, 5, 10, 25, 50], 100) 
 print(ways_dp)
```
数学上可以用**枚举法**
```python
#枚举做法,代码复杂度:O(n^k)
# 初始化方法数
count = 0

# 枚举1美分硬币的数量
for one_cents in range(101):  # 从0到100
    # 枚举5美分硬币的数量
    for five_cents in range((100 - one_cents) // 5 + 1):
        # 枚举10美分硬币的数量
        for ten_cents in range((100 - one_cents - 5 * five_cents) // 10 + 1):
            # 枚举25美分硬币的数量
            for twenty_five_cents in range((100 - one_cents - 5 * five_cents - 10 * ten_cents) // 25 + 1):
                # 计算剩余金额
                remaining = 100 - one_cents - 5 * five_cents - 10 * ten_cents - 25 * twenty_five_cents
                # 检查剩余金额是否是50美分的整数倍
                if remaining % 50 == 0:
                    # 如果是，则这是一种有效的组合方式
                    count += 1

# 输出结果
print(count)

```
# 丢番图方程概论
对于上面的问题我们往往抽象成一个不定方程,又叫做**丢番图方程**(Diophantine Equations).
```ad-info
title:丢番图
丢番图*(Diophantus)*,古希腊人,于B.C.300年在自己的《算术》中提到这种方程.
```
## 二元丢番图方程的求解
### 什么是二元丢番图方程
$$
ax+by=c
$$
一般被称作**二元线性丢番图方程**,其中$a,b,c\in \mathbb{Z}$,$x,y$是整数变量.
二元丢番图方程讨论的其实是线性函数是否通过**格点**的问题.
首先我们先研究一个简化的情形:
$$
ax+by=1
$$
并介绍一些方法
### 裴蜀(Bezout)定理和扩展欧几里得算法
对于上面的方程,我们有一**裴蜀定理(Bezout's Theorem)**
来描述:
```ad-abstract
title:Bezout's Theorem
若$a,b\in \mathbb{Z}$,且$\gcd(a,b)=1$,则$ax+by=1$有整数解.
```
#### 裴蜀定理的证明
证明基于**扩展欧几里得算法**
```ad-info
title:欧几里得算法
欧几里得算法是一种求两个整数的最大公约数的方法.
算法的基本思想是，对于任意两个整数 $a$ 和 $b$（其中 $a > b$），可以反复使用除法，即 $a = bq + r$，其中 $q$ 是商，$r$ 是余数，且 $0 \leq r < b$.
这个过程可以重复进行，直到余数为0。最后的非零余数就是 $a$ 和 $b$ 的最大公约数.
有的时候我们也叫它**辗转相除法**
```
### 扩展欧几里得算法来证明裴蜀定理
**扩展欧几里得算法**在欧几里得算法的基础上,通过记录每一步的商和余数,来找到满足裴蜀定理的整数 $x$ 和 $y$.具体步骤如下:
- 初始时，设 $r_0 = a$，$r_1 = b$，$s_0 = 1$，$s_1 = 0$，$t_0 = 0$，$t_1 = 1$。
- 重复执行以下步骤，直到 $r_i = 0$：
      - 计算 $q_i = \left\lfloor \frac{r_{i-2}}{r_{i-1}} \right\rfloor$
      - 更新 $r_i = r_{i-2} - q_i r_{i-1}$
      - 更新 $s_i = s_{i-2} - q_i s_{i-1}$
      - 更新 $t_i = t_{i-2} - q_i t_{i-1}$
- 当 $r_i = 0$ 时，$r_{i-1}$ 就是 $a$ 和 $b$ 的最大公约数，而 $s_{i-1}$ 和 $t_{i-1}$ 就是满足裴蜀定理的整数 $x$ 和 $y$.
通过扩展欧几里得算法，我们可以找到满足 $ax + by = \gcd(a, b)$ 的整数 $x$ 和 $y$。这是因为算法的每一步都在保持等式 $r_{i-2} = q_i r_{i-1} + r_i$ 的同时，更新 $s_i$ 和 $t_i$ 使得 $s_{i-1}a + t_{i-1}b = r_{i-1}$。当 $r_{i-1}$ 为最大公约数时，等式变为 $s_{i-1}a + t_{i-1}b = \gcd(a, b)$.

扩展欧几里得的代码实现如下:
```python
def extended_gcd(a, b):  
    if a == 0:  
        return b, 0, 1  
    else:  
        gcd, x1, y1 = extended_gcd(b % a, a)  
        x = y1 - (b // a) * x1  
        y = x1  
        return gcd, x, y  
  
# 测试代码  
a, b = 60, 48  
gcd, x, y = extended_gcd(a, b)  
print(f"gcd({a}, {b}) = {gcd}")  
print(f"{a} * {x} + {b} * {y} = {gcd}")
```
#### 二元丢番图方程的通解
二元丢番图方程的判定定理如下:
```ad-abstract
title:二元丢番图方程解的判定定理
对于二元丢番图方程,设$a,b$是整数,且$\gcd(a,b)=d$.若$d$不能整除$c$,那么就无解.
否则就有无穷多解,解形式为:
$$
\begin{aligned}
x=x_0+\cfrac{b}{d}n\\
y=y_0-\cfrac{a}{d}n
\end{aligned}
$$
也可以概括为:
**$ax+by=c$ $\Leftrightarrow$ $d=\gcd(a,b)$能整除$c$**
```
### 二元丢番图的方程的案例

```ad-question
title:线段上的格点数量
在二维平面上,给定两个格点$p_{1}(x_1,y_1)$和$p_{2}(x_2,y_2)$,问两点之间的线段上还有几个格点?(已知$x_1<x_2$)
```
很显然,我们先得将这个线段的方程求解出来:
$$
(y_{2}-y_{1})x-(x_{1}-x_{2})y=y_{2}x_{1}-y_{1}x_{2}
$$
于是我们有$a=y_{2}-y_{1},b=x_{1}-x_{2},c=y_{2}x_{1}-y_{1}x_{2},d=\gcd(a,b)=\gcd(\left| y_{2}-y_{1} \right|,\left| x_{1}-x_{2} \right|)$
于是我们根据通解的公式,可以得到:
$$
x=x_{0}+\frac{b}{d}n
$$
是一个通解.
现在我们有限制条件$x_{1}<x<x_{2}$
所以$x_{1}<x_{1}+\left( x_{1}-\cfrac{x_{2}}{d} \right)n<x_{2}$此时$-d<n<0$
即$n$有$d-1$个取值,即内有$d-1$格点.
## 多元丢番图方程的求解
**多元丢番图方程**比较复杂，这里仅给一个定理：
```ad-summary
title:多元丢番图方程的定理
$$
a_{1}x_{1}+a_{2}x_{2}+\cdots+a_{n}x_{n}=a
$$
其中$a$和$a_{i}$是整数.
则整数解存在当且仅当
$$
\gcd(a_1,a_2,\dots,a_n)=1
$$
```
# DP(动态规划)
## 引入和割棒问题
**DP(Dynamic Programming)**,俗称**动态规划**.是一种讲问题分解成子问题来求解的一种方法.这些方法之间有相互重复的子问题.DP在解决这些子问题的时候对结果进行了存储,从而不反复地做这些工作.
我们一般在处理**最优化问题**上使用DP,下面是一个案例：
```ad-todo
title:割棒问题
对于一根钢筋而言,割成不同长度的情况下卖出的钱也不一样,对于长为8钢筋,其不同长度切割的价格满足数组:[1, 5, 8, 9, 10, 17, 17, 20],求最大利润和最大利润的切割方案
```
```Python
#求利润值的代码
def cut_rod(prices, n):  
    # dp[i]表示长度为i的钢筋的最大利润  
    dp = [0] * (n + 1)  
  
    # 构建动态规划表  
    for i in range(1, n + 1):  
        max_profit = float('-inf')  # 设置一个初始的最小利润值  
        for j in range(i):  
            # 如果当前价格大于等于0，则考虑切割  
            if prices[j] + dp[i - j - 1] > max_profit:  
                max_profit = prices[j] + dp[i - j - 1]  
        dp[i] = max(max_profit, prices[i - 1] if i - 1 < len(prices) else 0)  
  
    return dp[n]  
  
  
# 钢筋的长度  
length = 8  
# 长度为1~n的钢筋的价格  
prices = [2,1,2,1,1,2,2,1]  
  
# 计算最大利润  
max_profit = cut_rod(prices, length)  
print("Maximum profit:", max_profit)
```
```python
def cut_rod(prices, n):  
    # dp[i]表示长度为i的钢筋的最大利润  
    dp = [0] * (n + 1)  
    # path记录最优切割方案  
    path = [0] * (n + 1)  
  
    for i in range(1, n + 1):  
        for j in range(len(prices)):  
            if j <= i:  
                profit = prices[j] + dp[i - j - 1] if i - j - 1 >= 0 else 0  
                if profit > dp[i]:  
                    dp[i] = profit  
                    path[i] = j  
  
    return dp[n], path  
  
  
def print_cut_rod(path, n):  
    result = []  
    i = n  
    while i > 0:  
        cut_len = path[i]  
        result.append(cut_len + 1)  # +1 to adjust for 0-based index  
        i -= cut_len + 1  # move to the previous cut position  
    print("Cut the rod at lengths:", ", ".join(map(str, result[::-1])))  
  
  
# 钢筋的长度  
length = 8  
# 长度为1~n的钢筋的价格  
prices = [1, 5, 8, 9, 10, 17, 17, 20]  
  
# 计算最大利润和最优切割位置  
max_profit, path = cut_rod(prices, length)  
print("Maximum profit:", max_profit)  
print_cut_rod(path, length)
```
此时我们看向那个割法(下面那个图是割长度为4的钢管,价值表是[1,5,8,9,10,17,17,20,24,30]):
![[Pasted image 20240917224906.png]]
很显然,我们能认识到,我们可以把割完的新钢管视为一个新的钢管切割问题.只要我们能够让这两个钢管分别达到最优情况,很显然我们也能得到切割的最优化情况.我们说，钢筋切割问题表现出**最优子结构**的特性,即问题的最优解包含了相关子问题的最优解，这些子问题我们可以独立解决.实际上,我们对于上面的问题,可以抽象出下面方程式:
$$
r_{n}=\max(p_{n},r_{1}+r_{n-1},r_{2}+r_{n-2},\dots,r_{n-1}+r_{1})
$$
再简洁一点,我们有
$$
r_{n}=\max_{1\leq i\leq n}(p_{i}+r_{n-i})
$$
如果我们已经决定将钢筋的第一部分切割成特定的长度，那么剩下的问题就只是如何最优地处理剩余的部分.在这种情况下，我们不需要考虑第一段的切割方案，因为我们已经做出了决定，现在只需要专注于剩余部分的最优解.
所以这个割棒问题的伪代码应该如下:
## 递归,但未DP优化

````ad-hint
title:伪代码
```python
def CUT_ROD(p,n)
    #数组形式的价格的表
    p[n]={p1,p2,...,pn}
    if n == 0 #长度为0
        return 0 #没办法切长度为0的棒子,当然价格也是0
    q = -Inf #取初始值为极小的
    #遍历整个n(也就是给定和式的范围),执行上面的式子
    for i=1 to n
        q = max(q,p[i]+CUT_ROD(p,n-i))
    #返回求得的值,就是最优解
    return q 
``` 
````
我们用Python实现一下
```Python
#标准错误代码
import sys  
def cut_rod(p,n):  
    if n==0:  
        return 0  
    q=-sys.float_info.max  
    for i in range(1,n):  
        q=max(q,p[i]+cut_rod(p,n-i))  
    return q  
price=[1,1,4,5,1,4]  
length=6  
max_price=cut_rod(price,length)  
print(max_price)
```
这个玩意的输出是$-1.7976931348623157e+308$
也就是我们利用sys包设置的最小值,这显然是错的,所以没写注释.

之所以是错的,是因为:
- $range(1,n)$是从$0$遍历到$n-1$位置,应该用$range(1,n+1)$
- 数组的索引从0开始,所以第一个遍历项应该是$p[i-1]$
```Python
#正确的代码
import sys  
def cut_rod(p,n):  
    if n==0:  
        return 0  
    q=-sys.float_info.max  
    for i in range(1,n+1):  
        q=max(q,p[i-1]+cut_rod(p,n-i))  
    return q  
price=[1,1,4,5,1,4]  
length=6  
max_price=cut_rod(price,length)  
print(max_price)
```
此时输出正确的答案:
![[Pasted image 20240917233346.png]]
我们以已经做过的割长度为4的钢管为例,得出答案为
![[Pasted image 20240917233419.png]]
所以也是正确的.

割棒问题可以认为是最简单的DP问题.利用这种递归的方式确实可以求出我们的最优金额,但是这个算法是**羸弱**的,当$length$为40时,电脑真的会寄,Python本来运行速度就很慢,这迫使缺少时间的肉夹馍去分析为何会这么慢.
实际上,想必大家注意力和理解力惊人,一下字就能想到上面的操作其实就是画了棵树:
![[Pasted image 20240917234031.png|350]]
假如你一下子想不出来(可能只是突然想不明白),我来解释下,这棵树(数据结构会教什么是树,现在你就想它是树状图).那么每一个树就会形成一个子树,也就是我们的子问题,所以实际上直到叶子节点它才会停止,所以它的次数
$$
T(n)=1+\sum^{n-1}_{j=0}T(j)
$$
如果你的树状图学的不错的话,那么我们就可以得出最坏的情况下,就是整棵树都被生成了,也就是有
$$
T(n)=2^n
$$
所以其实它是一个指数级的算法(很差):$O(n\cdot 2^n)$,这简直受不鸟.
## DP优化
所谓算法无非空间换时间/时间换空间,对于DP而言,它就是空间换时间.
### 自顶向下备忘录法
我们还是从头到尾递归这个问题,但是我们用一个**数组**或者一个**哈希表(Python中就是字典)**.假如我们这个子问题已经算过了,那么它就直接从字典里把这个值调出来用,而省去建一棵子树的麻烦.如果没算过,那当然还是要老老实实算的.
我们说这个递归方法是**记录化**(memo-ized)的.*而不是记忆化的(memo-rized)*.
自顶向下的伪代码如下
````ad-info
title:伪代码
```Python
def mem_cut_aux(p,n,r):
if r[n]>=0
    return r[n]
if n == 0
    q = 0
else q = -inf
    for i = 1 to n
        q=max(q,p[i]+mem_cut_aux(p,n-i,r))
r[n]=q
return q

def mem_cut_rod(p,n):
for i in range(0,n)
    r[i]=-inf
return Mem_cut_rod_aux(p,n,r)

```
````

我们用Python实现一下
```Python
import sys  
def mem_cut_aux(p,n,r):  
    if r[n] >= 0:  
        return r[n]  
    if n==0:  
        q = 0  
    else:  
        q = -sys.float_info.max  
        for i in range(1,n+1):  
            q=max(q,p[i-1]+mem_cut_aux(p,n-i,r))  
        r[n] = q  
    return q  
  
def mem_cut(p,n):  
    r = [-sys.float_info.max] * (n + 1)  
    return mem_cut_aux(p,n,r)  
print(mem_cut([1,1,4,5,1,4],6))
```
### 如何复原一个**俄罗斯套娃**??
比起我们一边写一边记,我们还有一种思路,那就是从小的开始算,一步一步拼成大的.我们将子问题按大小排列,先处理小的子问题,很明显大的子问题是小的子问题解答的组合,于是我们就像俄罗斯套娃一样一个一个装了回去.
````ad-info
title:伪代码
```Python
def bottom_up_cut_rod(p,n):
    r[0]=0
    for j = 1 to n
        q=-Inf
        for i = 1 to j
            q=max(q,p[i]+r[j-i])
        r[j]=q
    return r[n]
```
````
那我们现在用python写一下
```python
import sys  
  
def bottom_up_method(p, n):  
    r = [0] * (n + 1)  # 将数组长度扩展到 n+1 
   for j in range(1, n + 1):  
        q = -sys.float_info.max  
        for i in range(j):  
            if i < len(p):# 确保 p[i] 在 p 的范围内  
                q = max(q, p[i] + r[j - i - 1])  
        r[j] = q  
    return r[n]  
  
# 测试代码  
print(bottom_up_method([1, 1, 4, 5, 1, 4], 6))
```
## 我想知道这玩意他妈的怎么割开! :解的重构造
现在我们扩展一下我们上面的俄罗斯套娃法,使得它不仅能算出最优化的价格,还能算出我们的切割方法,下面是我们的伪代码
````ad-info
title:伪代码
```Python
#上面存利润下面存割法
r[n]={0,...,n}
s[n]={0,...,n}
def ex_bottom_up_method(p,n)
r[0]=0
for j = 1 to n
    q=-inf
    for i = 1 to j
         if q<p[i]+r[j-i]
         s[j]=i
    r[j]=q
return r and s
```
````
```Python
import sys  
  
def bottom_up_method(p, n):  
    r = [0] * (n + 1)  # 将数组长度扩展到 n+1    s = [0] * (n + 1)  # 用于保存选择的物品的决策  
    for j in range(1, n + 1):  
        q = -sys.float_info.max  
        for i in range(1, j + 1):  # i 的范围从 1 到 j            if i - 1 < len(p):  # 确保 p[i-1] 在 p 的范围内  
                if q < p[i - 1] + r[j - i]:  # 注意 r[j-i] 而不是 r[j-i-1]                    q = p[i - 1] + r[j - i]  
                    s[j] = i  # 记录切割位置  
        r[j] = q  
    # 回溯找出最优切割位置  
    cut = []  
    k = n  
    while k > 0:  
        cut.append(s[k])  # 添加切割位置  
        k -= s[k]  # 更新 k 为剩余部分  
    cut.reverse()  # 反转切割位置列表  
    return r, cut  
  
# 测试代码  
p = [1, 1, 4, 5, 1, 4]  # 每单位长度绳子的价值  
n = 6  # 绳子的总长度  
r, cut = bottom_up_method(p, n)  
print("最大价值:", r[n])  
print("切割位置:", cut)
```
## 练习
```ad-question
请用DP,给出一个斐波那契数列求解的O(n)算法.
```
普通的**Fibonacci数列**使用递归来计算:
```python
def fib(n):
    if n==1 or n==2:
        return 1
    else:
        return fib(n-1)+fib(n-2)      
```
很显然它的复杂度是$O(2^{n})$,计算机在计算$n=50$的时候就会吃不太消.
我们发现是因为前面的fib数被反复地计算导致的,这就促使我们使用DP中的**记忆化搜索**
```Python 
def fib_memo(n,memo = None):  
      if memo is None:#表示不存在这样的字典,也就是初始化字典的过程.
          memo={}#就创建一个字典以备不时之需.
      if n in memo:
          return memo[n]#如果我们已经算过,那么就从字典里拿数,免得再算
      if n == 1 or n == 2:
          return 1 #特殊情况特判.
    memo[n] = fib_memo(n-1,memo)+fib_memo(n-2,memo)#memo[n]的递推式.
    return memo[n]
#测试数据
print(fib(50))
```
这时很快算出了Fibonacci数列的第50项.
![[Pasted image 20240918141226.png]]
此时时间复杂度和空间复杂度都是$O(n)$,因为你填充memo这个哈希表的过程是线性的,所以它只和哈希表的长度,也就是$n$有关系.
问题出现在这里,现在我们$fib(5000)$:
![[Pasted image 20240918141723.png]]
会爆出这样的错,即超出递归深度.
这是python的问题,因为递归在python中很**昂贵**,深度超过1000就会报错,强行修改递归深度限制也是可以跑的,但是性能堪忧,我们能不能避免递归?等一下?制表??
其实我们只需要把第二项赋给第一项,第三项赋给第二项不就行了吗??也就是:
```python 
a,b=b,a+b
```
这不就不用递归了??
```python
#优化的自底向上DP
def fib_optimized(n):
    if n == 1 or n == 2:
        return 1
    a, b = 1, 1#直接特判的情况
    for i in range(3, n + 1):
        a, b = b, a + b#循环赋值,根本就不用记住前面的内容
    return b

# 测试代码
print(fib_optimized(5000))
```
此时就不会报错,并且我们有$$F_{5000}=3878968454388325633701916308325905312082127714646245106160597214895550139044037097010822916462210669479293452858882973813483102008954982940361430156911478938364216563944106910214505634133706558656238254656700712525929903854933813928836378347518908762970712033337052923107693008518093849801803847813996748881765554653788291644268912980384613778969021502293082475666346224923071883324803280375039130352903304505842701147635242270210934637699104006714174883298422891491273104054328753298044273676822977244987749874555691907703880637046832794811358973739993110106219308149018570815397854379195305617510761053075688783766033667355445258844886241619210553457493675897849027988234351023599844663934853256411952221859563060475364645470760330902420806382584929156452876291575759142343809142302917491088984155209854432486594079793571316841692868039545309545388698114665082066862897420639323438488465240988742395873801976993820317174208932265468879364002630797780058759129671389634214252579116872755600360311370547754724604639987588046985178408674382863125$$
我写在这,昭告天下:fib已经寄了.
## 用DP实现矩阵的链式乘法
### 矩阵链式乘法
现在假定我们有矩阵$A_{1},A_{2},A_{3}\dots A_{n}$
首先我们知道矩阵必须相容才能乘起来吧,用代码实现就是先得做相容性判定,然后带公式,伪代码如下:
````ad-info 
title:矩阵乘法的伪代码
```py
if A.columns!=B.rows:
    return -1
else: 
    double C[A.rows][B.columns]
    for i in range(A.rows)
        for j in range(B.columns)
            cij=0
            for k in (A.rows)
            cij+=aik*bkj
```
````
所谓的矩阵链式乘法,就是指必须以括号的形式来确定矩阵的乘法顺序,从而进行矩阵乘法的次数最少.所以所谓的确定矩阵的乘法就是在确定括号的结合数.在矩阵链乘法问题中，我们实际上并不进行矩阵的乘法运算.我们的目标只是为了确定一个乘法顺序，使得乘法的成本最低.通常，确定这个最优顺序所花费的时间，通过后来实际进行矩阵乘法运算时节省的时间得到了更多的回报(例如，只进行7500次标量乘法运算，而不是75000次).
有人问这样会不会改变乘法的值,并不会,并思考为什么?






























