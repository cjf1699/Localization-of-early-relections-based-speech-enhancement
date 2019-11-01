import numpy as np 
import warnings
import config as c
import torch
import matplotlib.pyplot as plt
import time


def peak_detection(data, plot):
    assert len(data.shape) == 1

    N = len(data)
    tmp = None
    peaks = []
    for i in range(N):
        if i - c.sigma < 0 or i + c.sigma > N:
            if i - c.sigma < 0:
                tmp = torch.cat((data[i - c.sigma:], data[:i+c.sigma]))
            else:
                tmp = torch.cat((data[i-c.sigma:], data[0:(i + c.sigma) % N + 1]))
        else:
            tmp = data[i-c.sigma:i+c.sigma]

        if data[i] == torch.max(tmp) and data[i] > c.thres:
            peaks.append(i)

    if len(peaks) == 0:
        warnings.warn('没有峰值！\n')
        if plot:
            plt.figure()
            plt.plot(data.cpu().detach().numpy())
            plt.savefig('abnormal.jpg')
            plt.close()
    else:
        if plot:
            plt.figure()
            plt.plot(data.cpu().detach().numpy())
            plt.savefig('normal' + time.asctime(time.localtime(time.time())) + '.jpg')
            plt.close()
    return peaks # 返回峰值的角度


def cal_recall(pred, label):
    # check pass
    if len(pred) * len(label) == 0:
        raise RuntimeError('标签或预测的列表为空！\n')

    err = [5, 10, 15]
    
    recall = np.zeros(len(err))
    for idx, wucha in enumerate(err):
        cnt = 0
        for item in label:
            for yuce in pred:
                diff = np.abs(item - yuce)
                if diff > 180:
                    diff = 360 - diff
                if diff <= wucha:
                    cnt += 1
                    break
        recall[idx] = cnt / len(label)

    return recall


def cal_precision(pred, label):
    # check pass
    if len(pred) * len(label) == 0:
        raise RuntimeError('标签或预测的列表为空！\n')
    err = [5, 10, 15]
    prec = np.zeros(len(err))
    for idx, wucha in enumerate(err):
        cnt = 0
        for yuce in pred:
            for item in label:
                diff = np.abs(item - yuce)
                if diff > 180:
                    diff = 360 - diff
                if diff <= wucha:
                    cnt += 1
                    break
        prec[idx] = cnt / len(pred)

    return prec
