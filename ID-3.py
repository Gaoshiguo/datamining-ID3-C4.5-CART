#先导入数据集，我们选用的是鸢尾花的数据集，可以通过sklearn导入数据集
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
#观察打印出来的数据集，我们可以看到data属性是一个150行四列的矩阵，其中每一行代表一个样本数据，也就是一朵花的详细数据
#四列分别代表的属性是“花萼长度”、“花萼宽度”、“花瓣长度”、“花瓣宽度”
#再观察target属性我们发现，是一个包含150个数字的一维列表，代表对应data每一条样本花的属性分类，在数据集中使用0/1/2来表示三种花
#对数据集有所了解之后，我们开始对数据及进行分类，先使用ID3算法，ID3算法是以信息熵增益作为最优化分属性，所以我们需要计算出以数据集中四个不同属性
#作为划分时的信息熵，选择信息熵增益最大的那个属性作为第一次划分的最优属性。具体操作方法就是，先计算未划分时的数据集的信息熵，再计算按照四个属性分别分类后的信息熵
#用初始的信息熵减每一个属性划分的信息熵，取其中差值最大的那个

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

#计算出初始的信息熵之后，我们再计算分别按照每一个属性分类后的信息熵
#先计算出以第一个属性花萼长度作为划分属性的信息熵：


#获取所有的第一个属性值
list_1 = []
for i in range(0,len(iris.data)):
    list_1.append(iris.data[i][0])
    i = i+1
print(list_1)#打印输出数据集中所有的一维属性值
#计算列表中相同属性值的个数
se = set(list_1)#使用集合来去除列表中重复的元素
se_list=list(se)#集合没有索引，将集合强制转换成列表
print(se_list)

#计算相同元素的个数
list_count = []
for i in range(0,len(se_list)):
    x = list_1.count(se_list[i])
    print(se_list[i],"共有",x,"个","其对应的编号是",)
    list_count.append(x)
    i = i+1

print(list_count)
a =zeros((150,2))
m=0
#将样本属性相同的值与其对应的花的种类存储在一个150行两列的二维数组中
for i in range(0,len(se_list)):
    for j in range(0,len(list_1)):
        if se_list[i]==list_1[j]:
            a[m]=(se_list[i],iris.target[j])
            j=j+1
            m=m+1

    i=i+1

print(a)
#定义计算各个信息熵的函数，将概率作为传入的参数
def calc_entropy(ps0,ps1,ps2):
    if ps0==1 or ps1==1 or ps2==1:
        Ent=0
    elif ps0==0:
        Ent= -(0 + ps1*math.log(ps1)/math.log(2) + ps2*math.log(ps2)/math.log(2))
    elif ps1==0:
        Ent = -(ps0 * math.log(ps0) / math.log(2) + 0 + ps2 * math.log(ps2) / math.log(2))
    elif ps2==0:
        Ent = -(ps0 * math.log(ps0) / math.log(2) + ps1 * math.log(ps1) / math.log(2) + 0)
    else:
        Ent = -(ps0 * math.log(ps0) / math.log(2) + ps1 * math.log(ps1) / math.log(2) + ps2 * math.log(ps2) / math.log(2))
    print(Ent)

calc_entropy(2/7,5/7,0)
#将每个花萼属性可能取值的最终种类概率计算出来，存入一个x行3列的矩阵中，x的取值取决于该属性的取值
#在花萼属性中，共有35个不同的样本数据，所以该矩阵为35x3
#定义函数计算概率
p0_count=0
p1_count=0
p2_count=0
k=0
for j in range(0, len(se_list)):
    for i in range(0,len(a)):
        if a[i][0] == se_list[j]:
            if a[i][1]==0:
                    p0_count = p0_count+1
            elif a[i][1]==1:
                    p1_count = p1_count+1
            else:
                    p2_count = p2_count+1
        i=i+1
        print(p0_count)
    j=j+1















