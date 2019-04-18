# datamining-ID3-C4.5-CART
这个项目用来展示经典的ID3/C4.5/CART算法，首先需要知道的是这三个算法的原理以及他们的不同之处。****__ID-3算法是依据信息熵增益作为划分的最优属性，C4.5是在ID-3的基础上依据信息熵增益率作为划分的最优划分属性，而CART算法是依据基尼系数作为划分的最优属性__****
## 一、数据集介绍
我们选用的是IRIS鸢尾花数据集，选择这个做数据集的原因是，这个数据集已经非常普遍被人所使用的，而且Python中的Sklearn包已经行包含了该数据集，可以直接调用。

### 1.1首先我们来看一下如何导入数据集和查看数据集

由于Sklearn数据集已经将iris数据集完整封装好，我们为了方便，只需要导入该包，直接调用即可。

具体代码参见下图：

```
from sklearn.datasets import load_iris
import pandas as pd
import math
import  numpy as np
from numpy import *
from numpy import array as matrix, arange
iris = load_iris()
print(iris.data)
print(iris.data.shape)
print(iris.target)
print(iris.target.shape)

```
![image](https://github.com/Gaoshiguo/datamining-ID3-C4.5-CART/blob/master/iris-image/1.png)

![image](https://github.com/Gaoshiguo/datamining-ID3-C4.5-CART/blob/master/iris-image/2.png)

从代码运行效果图中我们可以看到：`iris.data`是一个150行4列的矩阵，存储着鸢尾花的四个属性的样本信息值，从数据集的简介中，这四个属性分别是花萼长度、花萼宽度、花瓣长度、花瓣宽度。`iris.target`存储的是150个数据的一维向量，代表150个样本的花所属于0,1,2中的哪一类，分别用0,1,2来代表

对数据集有所了解之后，我们需要理解信息熵的概念，信息熵我所理解的就是信息的混乱程度、不确定程度。周志华的《机器学习》中对信息熵的计算有详细的介绍。如下图所示：

![image](https://github.com/Gaoshiguo/datamining-ID3-C4.5-CART/blob/master/iris-image/3.png)

所以，我们在计算信息增益时首先要做的就是先计算出整个数据集最初始时的信息熵，然后选择四个属性中的某一个属性作为最优划分属性，计算按照该属性划分后的信息熵，计算两者的差值，这个差值称之为信息增益，信息熵越小代表信息越纯，信息增益越大代表最优划分属性选择的越好，最容易划分数据集

## 二、计算初始信息熵

### 2.1首先我们来看一下如何计算初始信息熵

依据信息熵的定义，我们需要先计算三种花在整个数据集中分别所占的比率，然后用每种花的比率乘以log2（p），其中p代表每种花的概率。具体的实现代码如下：

```
#计算初始数据集的信息熵
#先计算三种花在整个数据集中分别所占的比率
p_0 = 0
p_1 = 0
p_2 = 0
sum = sum(iris.target)
for i in iris.target:
    if i==0:
        p_0 = p_0+1
    elif i==1:
        p_1 = p_1 + 1
    elif i==2:
        p_2 = p_2 + 1

print('品种为0的花有',p_0,"个")
print('品种为1的花有',p_1,"个")
print('品种为2的花有',p_2,"个")

#计算出每种花的个数之后，在算出每种花所占的比率

p0 = p_0/sum
p1= p_1/sum
p2 = p_2/sum
print("品种为0的花在样本中所占的比率为",p0,"%")
print("品种为1的花在样本中所占的比率为",p1,"%")
print("品种为2的花在样本中所占的比率为",p2,"%")

#计算出比率后，按照信息熵的计算公式计算出初始状态的信息熵
ent = -(p0*math.log(p0)/math.log(2) + p1*math.log(p1)/math.log(2) + p2*math.log(p2)/math.log(2))
print("初始的信息熵为：",ent)

```
运行结果图如下图所示：

![image](https://github.com/Gaoshiguo/datamining-ID3-C4.5-CART/blob/master/iris-image/4.png)


