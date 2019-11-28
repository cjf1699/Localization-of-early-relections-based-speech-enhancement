import matplotlib.pyplot as plt
import matplotlib
import numpy as np 

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

label_list = list(range(0, 360, 5))
temp1 = np.load('prec_rec_for_each_snr_10.npy', allow_pickle=True)
temp1 = temp1[()]

prec1 = temp1['p']
rec1 = temp1['r']
prec1 = prec1[0:72, 0] * 100
rec1 = rec1[0:72, 0] * 100
# num_list1 = [20, 30, 15, 35]
# num_list2 = [15, 30, 40, 20]
x = np.array(list(range(len(prec1)))) * 5
plt.figure(figsize=(80, 4))
plt.plot(x, prec1, color='blue', label="prec")
# rects2 = plt.bar(x=x, height=rec, width=0.45, color='green', label="recall", bottom=prec)
# plt.ylim(0, 100)
plt.ylabel("%")
plt.xticks(range(0,360,30))
plt.xlabel("deg")
plt.legend()
plt.show()
