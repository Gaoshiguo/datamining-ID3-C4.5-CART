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

对数据集有所了解之后，我们需要理解信息熵的概念，信息熵我所理解的就是信息的混乱程度、不确定程度。周志华的《机器学习》中对信息熵的计算有详细的介绍。
