from _datetime import datetime
import pandas as pd
import numpy as np
from dtw import dtw
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from PyEMD import EMD
import matplotlib

matplotlib.rc("font", family='DengXian')


class DTW():
    # # 调用DTW
    # mkt_9 = ['CSI300', 'HSI', 'STI', 'FCHI', 'GDAXI', 'FTSE', 'SPX', 'DJI', 'NDX']
    # df_dtw = DTW(pre_name=mkt_9)
    # df_corr = df_dtw.get_res()
    def __init__(self, pre_name):
        self.mkt_name = pre_name

    def plt_dtw(self, x, y):
        alignment = dtw(x, y, keep_internals=True)
        # alignment.plot(type="threeway")
        # alignment.plot(type="twoway")
        return -alignment.distance

    def get_mkt_name(self, path):  # 获取外国市场名称
        mkt_data = pd.read_csv(path)
        markets = list(mkt_data.columns)
        return markets[1:]

    def get_raw_data(self):
        # 获取股市数据
        dom_mkt = pd.read_csv("./raw_data/Daily Market Information of Domestic Index.csv", dtype={"IndexCode": str})
        dom_mkt = dom_mkt[dom_mkt["IndexCode"] == '000300']
        for_mkt = pd.read_csv("./raw_data/Daily Market Foreign Index.csv")
        for_mkt_clean = for_mkt[(for_mkt["Trddt"] > '2020-01-01') & (for_mkt["Trddt"] < '2020-06-01')]
        dom_mkt_clean = dom_mkt[(dom_mkt["Date"] > '2020-01-01') & (dom_mkt["Date"] < '2020-06-01')]
        # 处理
        all_data = pd.DataFrame()
        for i in self.mkt_name:
            if i == 'CSI300':
                dom_mkt_clean.reset_index(inplace=True)
                tmp = dom_mkt_clean["CloseIndex"]
            else:
                ttmp = for_mkt_clean[for_mkt_clean["Indexcd"] == i]
                ttmp.reset_index(inplace=True)
                tmp = ttmp["Clsidx"]
                # tmp.reset_index(inplace=True)
            all_data[i] = tmp
        scaler = MinMaxScaler()
        clean_data = scaler.fit_transform(all_data)
        clean_data = pd.DataFrame(clean_data, columns=all_data.columns)
        return clean_data

    def get_res(self):
        # path = "./raw_data/all cross market data.csv"
        # mkt_name = self.get_mkt_name(path)
        raw_data = self.get_raw_data()
        # dtw
        df_corr = raw_data.corr(method=self.plt_dtw)
        df_corr.to_csv("./output/corr/dtw.csv")
        plt.subplots(figsize=(5, 5))
        sns.heatmap(df_corr, cmap='Blues', annot=True)
        plt.show()
        return df_corr


def corrl():
    # 无参数，直接调用函数
    def euc(vec1, vec2):
        return -np.sqrt(np.sum(np.square(vec1 - vec2)))

    df_raw = pd.read_csv("./raw_data/all cross market data.csv")
    mkt = ['CSI300', 'HSI', 'STI', 'FCHI', 'GDAXI', 'FTSE', 'SPX', 'DJI', 'NDX']
    df = df_raw[mkt]
    # norm
    scaler = MinMaxScaler()
    clean_data = scaler.fit_transform(df)
    clean_data = pd.DataFrame(clean_data, columns=df.columns)
    # calc
    df_corr = clean_data.corr(method='spearman')
    df_corr.to_csv("./output/corr/spearman.csv")
    df_corr1 = clean_data.corr(method=euc)
    df_corr1.to_csv("./output/corr/euc.csv")
    # 画图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_corr, annot=True, cmap="YlGnBu")
    plt.title("Spearman Correlation", y=-0.2, fontsize=13)
    plt.subplot(1, 2, 2)
    sns.heatmap(df_corr1, annot=True, cmap='YlGnBu')
    plt.title("Euclidean Distance", y=-0.2, fontsize=13)
    plt.show()
    return None


def get_tdcc(mkt_list):
    # # 调用函数示例
    # mkt_list = ['HSI', 'STI', 'FCHI', 'GDAXI', 'FTSE', 'SPX', 'DJI', 'NDX']
    # get_tdcc(mkt_list)
    def crosscorr(datax, datay, lag=0, wrap=False):
        """ Lag-N cross correlation.
        Shifted data filled with NaNs

        Parameters
        ----------
        lag : int, default 0
        datax, datay : pandas.Series objects of equal length

        Returns
        ----------
        crosscorr : float
        """
        if wrap:
            shiftedy = datay.shift(lag)
            shiftedy.iloc[:lag] = datay.iloc[-lag:].values
            return datax.corr(shiftedy)
        else:
            return datax.corr(datay.shift(lag), method='spearman')

    df_raw = pd.read_csv("./raw_data/all cross market data.csv")

    # 滑动窗口时间滞后互相关
    def coldata(i):
        window_size = 100  # 样本
        t_start = 0
        t_end = t_start + window_size
        step_size = 20
        rss = []
        while t_end < len(df_raw):
            d1 = df_raw['CSI300'].iloc[t_start:t_end]
            d2 = df_raw[i].iloc[t_start:t_end]
            rs = [crosscorr(d1, d2, lag, wrap=False) for lag in range(-10, 10)]
            rss.append(rs)
            t_start = t_start + step_size
            t_end = t_end + step_size
        rss = pd.DataFrame(rss)
        return rss

    f, axs = plt.subplots(3, 3, figsize=(10, 10))
    f.suptitle('滑动窗口的滞后时间序列相关热力图', fontsize=16, weight='bold')
    xticks = np.arange(21)
    xlabel = '滞后阶数'
    ylabal = '时间窗口'
    # 0,0
    rss = coldata(mkt_list[0])
    axs[0, 0].imshow(rss, cmap='RdBu_r', aspect='auto')
    axs[0, 0].set(title='Table 1. CSI300 AND CSI300',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[0, 0].set_xticks(xticks)
    axs[0, 0].set_xticklabels([int(item - 10) for item in axs[0, 0].get_xticks()])
    # 0,1
    rss0 = coldata(mkt_list[1])
    axs[0, 1].imshow(rss0, cmap='RdBu_r', aspect='auto')
    axs[0, 1].set(title='Table 2. CSI300 AND HSI',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[0, 1].set_xticks(xticks)
    axs[0, 1].set_xticklabels([int(item - 10) for item in axs[0, 1].get_xticks()])
    # 0,2
    rss1 = coldata(mkt_list[2])
    axs[0, 2].imshow(rss1, cmap='RdBu_r', aspect='auto')
    axs[0, 2].set(title='Table 3. CSI300 AND STI',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[0, 2].set_xticks(xticks)
    axs[0, 2].set_xticklabels([int(item - 10) for item in axs[0, 2].get_xticks()])
    # 1,0
    rss2 = coldata(mkt_list[3])
    axs[1, 0].imshow(rss2, cmap='RdBu_r', aspect='auto')
    axs[1, 0].set(title='Table 4. CSI300 AND FCHI',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[1, 0].set_xticks(xticks)
    axs[1, 0].set_xticklabels([int(item - 10) for item in axs[1, 0].get_xticks()])
    # 1,1
    rss3 = coldata(mkt_list[4])
    axs[1, 1].imshow(rss3, cmap='RdBu_r', aspect='auto')
    axs[1, 1].set(title='Table 5. CSI300 AND GDAXI',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[1, 1].set_xticks(xticks)
    axs[1, 1].set_xticklabels([int(item - 10) for item in axs[1, 1].get_xticks()])
    # 1,2
    rss4 = coldata(mkt_list[5])
    axs[1, 2].imshow(rss4, cmap='RdBu_r', aspect='auto')
    axs[1, 2].set(title='Table 6. CSI300 AND FTSE',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[1, 2].set_xticks(xticks)
    axs[1, 2].set_xticklabels([int(item - 10) for item in axs[1, 2].get_xticks()])
    # 2,0
    rss5 = coldata(mkt_list[6])
    axs[2, 0].imshow(rss5, cmap='RdBu_r', aspect='auto')
    axs[2, 0].set(title='Table 7. CSI300 AND SPX',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[2, 0].set_xticks(xticks)
    axs[2, 0].set_xticklabels([int(item - 10) for item in axs[2, 0].get_xticks()])
    # 2,1
    rss6 = coldata(mkt_list[7])
    axs[2, 1].imshow(rss6, cmap='RdBu_r', aspect='auto')
    axs[2, 1].set(title='Table 8. CSI300 AND DJI',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[2, 1].set_xticks(xticks)
    axs[2, 1].set_xticklabels([int(item - 10) for item in axs[2, 1].get_xticks()])
    # 2,2
    rss7 = coldata(mkt_list[8])
    axs[2, 2].imshow(rss7, cmap='RdBu_r', aspect='auto')
    axs[2, 2].set(title='Table 9. CSI300 AND NDX',
                  xlim=[0, 20], xlabel=xlabel, ylabel=ylabal)
    axs[2, 2].set_xticks(xticks)
    axs[2, 2].set_xticklabels([int(item - 10) for item in axs[2, 2].get_xticks()])
    plt.show()


def get_emdc(mkt_list):
    # 调用函数示例
    # mkt_list = ['CSI300', 'HSI', 'STI', 'FCHI', 'GDAXI', 'FTSE', 'SPX', 'DJI', 'NDX']
    # out = get_emdc(mkt_list)
    # out.to_csv("./output/corr/emd_corr.csv")
    def convert_imfs(df, name):
        """
        调用EMD包，将原始数据进行EMD分解，分解为多个IMFS
        :param df: 原始数据
        :return: 原始IMFS数据
        """
        df.reset_index(drop=True, inplace=True)
        emd = EMD()
        s = np.array(df[name])
        t = np.arange(len(df))
        imfs = emd(s, t)
        columns = [i for i in range(len(imfs), 0, -1)]
        imfs = pd.DataFrame(np.transpose(imfs), columns=columns)
        return imfs

    def calc_corr(df1, df2):
        dist = 0
        for n in range(df1.shape[1]):
            dist += np.sum(np.square(df1[int(n + 1)] - df2[int(n + 1)]))
        return -round(np.sqrt(dist), 2)

    # 数据准备
    df_raw_ = pd.read_csv("./raw_data/all cross market data.csv")
    mkt = ['CSI300', 'HSI', 'STI', 'FCHI', 'GDAXI', 'FTSE', 'SPX', 'DJI', 'NDX']
    df_raw = df_raw_[mkt]
    scaler = MinMaxScaler()
    df_ = scaler.fit_transform(df_raw)
    df = pd.DataFrame(df_, columns=df_raw.columns)
    # 外部循环
    out_df = pd.DataFrame()
    for j in mkt_list:
        imfs_base = convert_imfs(df, j)
        imfs_base_use = imfs_base.iloc[:, -4:]
        # 保存结果
        res = []
        # 循环运算
        for i in mkt_list:
            imfs_i = convert_imfs(df, i)
            imfs_i_use = imfs_i.iloc[:, -4:]
            dist = calc_corr(imfs_base_use, imfs_i_use)
            res.append(dist)
        out_df[j] = pd.Series(res, index=mkt_list)
    # 画图
    plt.subplots(figsize=(5, 5))
    sns.heatmap(out_df, cmap='Blues', annot=True)
    plt.show()
    return out_df


def reshape_market():
    # 清洗HSI和SPX数据
    dmkt = pd.read_csv("./raw_data/Daily Market Information of Domestic Index.csv")
    fmkt = pd.read_csv("./raw_data/Daily Market Foreign Index.csv", dtype=object)
    new_fmkt_ = fmkt[fmkt["Trddt"] > '2010-01-04']  # 仅取用2010年之后的数据
    new_fmkt = new_fmkt_.copy(deep=True)
    new_fmkt["date"] = pd.to_datetime(new_fmkt["Trddt"])
    new_fmkt['dtr'] = new_fmkt["date"].dt.date
    # 准备指标数据
    new_code = ["HSI", "SPX"]
    # csi300
    h300_raw = dmkt[dmkt['IndexCode'] == 300]
    h300_raw.reset_index(inplace=True, drop=True)
    h300 = h300_raw.copy(deep=True)
    h300["Date_f"] = pd.to_datetime(h300["Date"])
    h300["dt"] = h300["Date_f"].dt.date

    # 关联
    for i in new_code:
        if i == 'HSI':
            tmp_table = new_fmkt[new_fmkt['Indexcd'] == i]
            # 筛选字段
            tmp_table_smp = tmp_table[["Indexcd", "dtr", "Opnidx"]]  # 临时表
            # 匹配，循环自联结
            h300 = h300.merge(tmp_table_smp, left_on='dt', right_on='dtr', how='left')
            # 规范输出
            h300.rename(columns={'Opnidx': i}, inplace=True)
            h300.drop(["Indexcd", "dtr"], axis=1, inplace=True)
        else:
            tmp_table = new_fmkt[new_fmkt['Indexcd'] == i]
            # 筛选字段
            tmp_table_smp = tmp_table[["Indexcd", "dtr", "Clsidx"]]  # 临时表
            # 匹配，循环自联结
            h300 = h300.merge(tmp_table_smp, left_on='dt', right_on='dtr', how='left')
            # 规范输出
            h300.rename(columns={'Clsidx': i}, inplace=True)
            h300.drop(["Indexcd", "dtr"], axis=1, inplace=True)

    # 替换
    # 深拷贝，数据切片容易报错
    final_amkt = h300.copy(deep=True)
    # 上文相同日期未匹配上，采用之前最近的交易日数据
    new_fmkt['dtr'] = new_fmkt['dtr'].astype(str)
    for i in new_code:
        if i == 'HSI':
            nan_date = list(final_amkt[final_amkt[i].isna()]["Date"])
            for j in nan_date:
                try:
                    df_new = new_fmkt[
                        (new_fmkt['Indexcd'] == i) & (new_fmkt['dtr'] < j)]["Opnidx"].iloc[-1]
                    mask = final_amkt['Date'] == j
                    tmp_index = final_amkt[mask].index
                    final_amkt.loc[tmp_index, i] = df_new
                except IndexError:
                    next
        else:
            nan_date = list(final_amkt[final_amkt[i].isna()]["Date"])
            for j in nan_date:
                try:
                    df_new = new_fmkt[
                        (new_fmkt['Indexcd'] == i) & (new_fmkt['dtr'] < j)]["Clsidx"].iloc[-1]
                    mask = final_amkt['Date'] == j
                    tmp_index = final_amkt[mask].index
                    final_amkt.loc[tmp_index, i] = df_new
                except IndexError:
                    next
    output = final_amkt.drop(columns=['Date_f', 'dt'])
    output.to_csv('./raw_data/SH300_with_hsi_spx.csv')
    return output


if __name__ == '__main__':
    pass
