import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import os


in_file = '../../data/'
out_file = '../../data_per_plane_new/'
listfile = os.listdir('../../data')
plane = ['P123', 'P124', 'P125', 'P126', 'P127']
columnName = ["时间", "全机油量", "武器重量", "马赫数", "气压高度", "校准空速", "真空速", "升降速度", "攻角", "侧滑角", "动压", "法向过载", "侧向过载", "轴向过载",
              "俯仰角", "滚转角", "航向角", "滚转速率", "俯仰速率", "偏航速率", "左鸭翼", "右鸭翼", "左前襟", "右前襟", "左外副翼", "右外副翼", "左内副翼", "右内副翼",
              "左方向舵", "右方向舵", "机翼剪力", "机翼弯矩", "鸭翼剪力", "垂尾剪力", "鸭翼弯矩", "机身弯矩", 'None']
for p in plane:

    DATA, M, S, number = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    for i in listfile:

        if i.startswith(p):
            data = pd.read_table(in_file + i, sep=' ', encoding='utf-8', names=columnName)

            # 丢掉最后nan列，去掉校准空速
            data = data.drop(['None', '校准空速'], axis=1)

            # 新数据1 全机重量，19353为空机重量
            data.insert(0, '全机重量', data['全机油量'] + data['武器重量'])
            data.iloc[0, 0] = data.iloc[0, 0] + 19353
            # 丢弃原重量数据
            data = data.drop(['全机油量', '武器重量'], axis=1)

            # 新数据2 加速度
            a = (data['俯仰速率'][3:].reset_index(drop=True) - data['俯仰速率'][2:-1].reset_index(drop=True)) * (
                        data['俯仰速率'][1] + 1000) / (
                            (data['时间'][3:].reset_index(drop=True) - data['时间'][2:-1].reset_index(drop=True)) * (
                                data['时间'][1] + 1000))
            a = pd.Series([-1000.0, -1000.0]).append(a).reset_index(drop=True)
            data.insert(20, '俯仰角加速度', a)
            a = (data['滚转速率'][3:].reset_index(drop=True) - data['滚转速率'][2:-1].reset_index(drop=True)) * (
                        data['滚转速率'][1] + 1000) / (
                            (data['时间'][3:].reset_index(drop=True) - data['时间'][2:-1].reset_index(drop=True)) * (
                                data['时间'][1] + 1000))
            a = pd.Series([-1000.0, -1000.0]).append(a).reset_index(drop=True)
            data.insert(21, '滚转角加速度', a)
            a = (data['偏航速率'][3:].reset_index(drop=True) - data['偏航速率'][2:-1].reset_index(drop=True)) * (
                        data['偏航速率'][1] + 1000) / (
                            (data['时间'][3:].reset_index(drop=True) - data['时间'][2:-1].reset_index(drop=True)) * (
                                data['时间'][1] + 1000))
            a = pd.Series([-1000.0, -1000.0]).append(a).reset_index(drop=True)
            data.insert(22, '偏航加速度', a)
            # 丢掉时间列
            data = data.drop(['时间'], axis=1)
            # 丢弃最后一行
            data.drop([len(data) - 1], inplace=True)

            # 均值
            m = data[0:1]
            m.insert(0, 'number', data.shape[0])
            M = pd.concat([M, m], ignore_index=True)

            # 标准差
            s = data[1:2]
            s.insert(0, 'number', data.shape[0])
            S = pd.concat([S, s], ignore_index=True)

            # 新数据
            data = data.drop([0, 1]).reset_index(drop=True)
            DATA = pd.concat([DATA, data], ignore_index=True)

    for col in data.columns:
        M[col] = M[col] * M['number']
        S[col] = S[col] * S['number']
    M = pd.DataFrame(np.array(M.sum().tolist()).reshape(1, M.shape[1]), columns=M.columns)
    S = pd.DataFrame(np.array(S.sum().tolist()).reshape(1, S.shape[1]), columns=S.columns)
    for col in data.columns:
        M[col] = M[col] / M['number']
        S[col] = S[col] / M['number']
    M = M.drop(['number'], axis=1)
    S = S.drop(['number'], axis=1)
    M = pd.concat([M, S], ignore_index=True)

    DATA.to_csv(out_file + p + '_data.csv', index=False, sep=',')
    M.to_csv(out_file + p + '_mean_sd.csv', index=False, sep=',')

    print(p + ' data file done')