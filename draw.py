import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# 设置风格，seaborn有5种基本风格，context表示环境
sns.set(style="darkgrid", context="talk")
# 处理中文问题
sns.set_style('whitegrid', {'font.sans-serif':['simhei', 'Arial']})


def draw_polar(x, y, fig_size=None, title=None):
    """
    draw a single polar picture
    :param x:
    :param y:
    :return:
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(x, y)
    plt.savefig(title if title else ('Untitled' + time.asctime(time.localtime(time.time()))) + '.jpg')
    # plt.show()


def draw(x, y, fig_size=None, title=None):
    """

    :param data: data to be plotted
    :return:
    """
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    if title is not None:
        plt.title(title)
    # plt.show()
    plt.savefig(title if title else ('Untitled' + time.asctime(time.localtime(time.time()))) + '.jpg')


def draw_v2(x, y1, y2, fig_size=None, title=None):
    fig = plt.figure(figsize=fig_size)
    ax1 = fig.add_subplot(211)
    ax1.plot(x, y1, 'r')
    ax2 = fig.add_subplot(212)
    ax2.plot(x, y2, 'b')
    if title is not None:
        plt.title(title)
    plt.savefig(title if title else ('Untitled' + time.asctime(time.localtime(time.time()))) + '.jpg')


def draw_bar(x, y, fig_size=None, title=None):
    fig = plt.figure(figsize=fig_size)
    sns.barplot(x, y, palette="BuPu_r")
    if not title is None:
        plt.title(title)
    plt.savefig(title if title else ('Untitled' + time.asctime(time.localtime(time.time()))) + '.jpg')
    # sns.despine(bottom=True)
    # plt.show()
