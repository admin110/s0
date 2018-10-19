import numpy as np
import matplotlib.pyplot as plt

x = np.array([123,150,87,102,280,200]).astype(np.float32)
y = np.array([250,320,160,220,590,390]).astype(np.float32)

# 参数设置
plt.rcParams['font.sans-serif']=['SimHei'] 	#用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False 	#用来正常显示负号

x_max = max(x)
x_min = min(x)
y_max = max(y)
y_min = min(y)

#生成新的x,y
for i in range(0,len(x)):					#range(stop)  range(start, stop, step)
	x[i] = (x[i] - x_min)/(x_max - x_min)
	y[i] = (y[i] - y_min)/(y_max - y_min)
	print("x[",i,"]=",x[i],",y[",i,"]=",y[i])

print("x=",x,"\n")
a = 1	
b = 0
x_ = np.array([0,1])
y_ = a*x_+b
yp = a*x +b
r = sum(np.square(np.round(yp-y,4)))	#round 返回浮点数y的四舍五入值
print("r =",r/10,"\n")

plt.scatter(x,y)						#坐标轴的取值范围	
plt.xlabel(u"价格")
plt.ylabel(u"面积")
plt.plot(x_,y_,color='green')				#x_ y_的连线
plt.plot([0,0.5],[0.5,0.5],color='red')		
plt.plot([0.5,0.5],[0,0.5],color='red')	
plt.scatter(0.5,0.5,color='red')
plt.pause(20)							#显示时间

