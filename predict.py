import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from math import sin, asin, cos, radians, fabs, sqrt

# 设定为自增长
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def mse(predictions, targets):
    return ((predictions - targets) ** 2).mean()


def reshape_y_hat(y_hat, dim):
    re_y = []
    i = 0
    while i < len(y_hat):
        tmp = []
        for j in range(dim):
            tmp.append(y_hat[i + j])
        i = i + dim
        re_y.append(tmp)
    re_y = np.array(re_y, dtype='float64')
    return re_y


# 数据切分
def data_set(dataset, test_num):  # 创建时间序列数据样本
    dataX, dataY = [], []
    for i in range(len(dataset) - test_num + 1):
        a = dataset[i:(i + test_num)]
        dataX.append(a)
        dataY.append(dataset[i + test_num - 1])
    return np.array(dataX), np.array(dataY)


# 多维反归一化
def FNormalizeMult(data, normalize):
    data = np.array(data, dtype='float64')
    # 列
    for i in range(0, data.shape[1]):
        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow
        print("listlow, listhigh, delta", listlow, listhigh, delta)
        # 行
        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = data[j, i] * delta + listlow

    return data


# 使用训练数据的归一化
def NormalizeMultUseData(data, normalize):
    for i in range(0, data.shape[1]):

        listlow = normalize[i, 0]
        listhigh = normalize[i, 1]
        delta = listhigh - listlow

        if delta != 0:
            for j in range(0, data.shape[0]):
                data[j, i] = (data[j, i] - listlow) / delta

    return data


EARTH_RADIUS = 6371  # 地球平均半径，6371km


# 计算两个经纬度之间的直线距离
def hav(theta):
    s = sin(theta / 2)
    return s * s


def get_distance_hav(lat0, lng0, lat1, lng1):
    # "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)

    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance


if __name__ == '__main__':
    test_num = 6
    per_num = 1
    ex_data = pd.read_csv('Geolife_Trajectories/Data/000/Trajectory/20081023025304.plt', sep=',', skiprows=6,
                          header=None).iloc[:, 0:2].values  # 原始数据
    data, dataY = data_set(ex_data, test_num)
    # data.dtype = 'float64'
    # y = dataY
    # #归一化
    normalize = np.load("./traj_model_trueNorm.npy")
    data_norm = []
    for i in range(len(data)):
        data[i] = list(NormalizeMultUseData(data[i], normalize))
        data_norm.append(data[i])

    model = load_model("./traj_model_120.h5")
    y_hat = []
    for i in range(len(data)):
        test_X = data_norm[i].reshape(1, data_norm[i].shape[0], data_norm[i].shape[1])
        dd = model.predict(test_X)
        dd = dd.reshape(dd.shape[1])
        dd = reshape_y_hat(dd, 2)
        dd = FNormalizeMult(dd, normalize)
        dd = dd.tolist()
        y_hat.append(dd[0])
    y_hat = np.array(y_hat)

    # 画测试样本数据库
    plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
    print(len(y_hat))
    print(len(ex_data))
    p1 = plt.scatter(ex_data[:, 1], ex_data[:, 0], c='r', marker='o', s=0.5, label='识别结果')  # 原始轨迹
    p2 = plt.scatter(y_hat[:, 1], y_hat[:, 0], c='b', marker='o', s=0.5, label='预测结果')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.grid()
    plt.show()
