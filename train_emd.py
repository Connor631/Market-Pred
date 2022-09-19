import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
import argparse
from multiprocessing import Pool
from functools import partial
import csv
from PyEMD import EMD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import GruModel
from model import TcnModel
from model import LSTMModel
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
warnings.filterwarnings("ignore")
np.random.seed(0)
matplotlib.rc("font", family='SimSun')


class ModelTraining:
    @staticmethod
    def train_lstm(train_x, train_y, test_x, args):
        """
        构建训练GRU模型。生成模型实例，调用训练模型并返回预测值。
        :param train_x: 训练集X
        :param train_y: 训练集y
        :param test_x: 测试集X
        :param args: 参数
        :return: 对测试集X的预测值
        """
        lstm = LSTMModel(batch_size=args.batch_size,
                         epoch=args.epoch,
                         seq_len=args.model_training_data_len,
                         lstm_cell_1=args.lstm_cell_1,
                         lstm_cell_2=args.lstm_cell_2)
        pred = lstm.train_model(train_x, train_y, test_x)
        return np.array(pred).reshape(1, -1)

    @staticmethod
    def train_gru(train_x, train_y, test_x, args):
        """
        构建训练GRU模型。生成模型实例，调用训练模型并返回预测值。
        :param train_x: 训练集X
        :param train_y: 训练集y
        :param test_x: 测试集X
        :param args: 参数
        :return: 对测试集X的预测值
        """
        gru = GruModel(batch_size=args.batch_size,
                       epoch=args.epoch,
                       seq_len=args.model_training_data_len,
                       gru_cell_1=args.gru_cell_1,
                       gru_cell_2=args.gru_cell_2)
        pred = gru.train_model(train_x, train_y, test_x)
        return np.array(pred).reshape(1, -1)

    @staticmethod
    def train_tcn(train_x, train_y, test_x, args):
        """
        构建训练TCN模型。生成模型实例，调用训练模型并返回预测值。
        :param train_x: 训练集X
        :param train_y: 训练集y
        :param test_x: 测试集X
        :param args: 参数
        :return: 对测试集X的预测值
        """
        tcn = TcnModel(nb_filters=args.tcn_nb_filters,
                       kernel_size=2,
                       epoch=args.tcn_epoch,
                       nb_stacks=1,
                       batch_size=args.tcn_batch_size,
                       nb_dilation=3
                       )
        pred = tcn.train_model(train_x, train_y, test_x)
        return np.array(pred).reshape(1, -1)


class DataProcess:
    @staticmethod
    def initial_file(args):
        """
        定义文件的保存位置，考虑续传的情况
        :param args: 预先定义参数中的file_name
        :return: none
        """
        # 记录模型结果
        if args.starting_point != 0:  # 续传
            for root, dirs, files in os.walk(".", topdown=False):
                if root == ".\output":
                    file_name = files[-1]
                    args.file_name = file_name
        else:
            path = ("./output/output_" + str(int(args.file_name)) + ".csv")
            with open(path, 'a', newline='') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(('lstm_pred', 'gru_pred', 'tcn_pred', 'target'))
        return None

    @staticmethod
    def prep_basic_data(code):
        """
        加载原始数据
        :param code: 种类代码
        :return: 筛选后的种类代码
        """
        MarketIndex = pd.read_csv(r"./raw_data/Daily Market Information of Domestic Index.csv"
                                  , dtype={"IndexCode": str})
        code_data = MarketIndex[MarketIndex["IndexCode"] == code]
        return code_data

    @staticmethod
    def convert_imfs(df):
        """
        调用EMD包，将原始数据进行EMD分解，分解为多个IMFS
        :param df: 原始数据
        :return: 分解后的IMFS数据
        """
        df.reset_index(drop=True, inplace=True)
        emd = EMD()
        s = np.array(df["CloseIndex"])
        t = np.array(df["Date"])
        imfs = emd(s, t)
        columns = ["emd_{}".format(i) for i in range(len(imfs))]
        imfs = pd.DataFrame(np.transpose(imfs), columns=columns)
        return imfs

    @staticmethod
    def get_window_data(data, i, interval):
        """
        将原始数据按列转化为监督型训练所需要格式，即将时间序列转化为窗口类型数据
        :param: data: 原始全量数据集
                i: 进行的第i次预测
                interval: 每次训练窗口采用的数据量，由于数据开始时间固定，实际用于指定牛市或熊市
        :return: 进行第i次预测所需要的数据,以及该预测的目标值。
        """
        target = data["CloseIndex"].iloc[i + interval]  # 所有训练集的下1条数据
        df_new = data.iloc[i:i + interval, :]  # 包下不包上
        return df_new, target

    @staticmethod
    def convert_sup(df, seq_len, batch_size):
        """
        将输入数据转化为监督型数据。
        :param df: 原始数据
        :param seq_len: 每条样本数据的长度
        :param batch_size: batch_size用于GRU和LSTM模型中
        :return: 训练集X，训练集y，测试集test_X
        """
        data_x = []
        data_y = []
        num = len(df) - seq_len
        for i in range(num % batch_size - 1, num):  # 保证总样本量可整除batch_size
            window = df.iloc[i:i + seq_len, :]
            if i == len(df) - seq_len - 1:  # 最后一条
                test_x = window.iloc[1:, :]  # 最后1天的数据作为预测测试
            else:
                x = window.iloc[:-1, :].to_numpy()
                y = window.iloc[-1, -1]
                data_x.append(x)
                data_y.append(y)
        # 统一格式
        x_out = np.array(data_x)
        y_out = np.array(data_y)
        test_x_out = test_x.to_numpy()
        return x_out, y_out, test_x_out

    @staticmethod
    def eval_all(df1, target, model, disp_cm=False, ret_info=False):
        """
        评估结果
        :param df1: 模型预测值
        :param target: 真实值
        :param model: 模型的名称
        :param disp_cm: 是否现实混肴矩阵
        :param ret_info: 是否画累计收益图
        :return: 模型的收益数据
        """
        mae = mean_absolute_error(df1, target)  # 绝对离差值
        rmse = np.sqrt(mean_squared_error(df1, target))  # 均方根误差
        r2 = r2_score(df1, target)
        mape = np.mean(np.abs((target - df1) / target))
        # 计算方向精度
        df_tmp = pd.DataFrame()
        df_tmp["sign"] = np.sign(df1 - target.shift())
        df_tmp["target_sign"] = np.sign(target - target.shift())
        df_tmp_sign = df_tmp[["target_sign", "sign"]]
        df_tmp_sign.dropna(inplace=True)
        # 构建混肴矩阵
        cm = confusion_matrix(df_tmp_sign["target_sign"], df_tmp_sign["sign"])
        if disp_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title("{} model confusion matrix".format(model))
            plt.show()
        tn, fp, fn, tp = cm.ravel()
        accu = round((tp + tn) / (tp + tn + fp + fn), 4)
        prec = round(tp / (tp + fp), 4)
        f1 = round(2 * tp / (2 * tp + fp + fn), 4)
        # 计算投资收益
        df_tmp["pos_s"] = np.where(df1 > target.shift(), 1, -1)  # 做多，做空
        df_tmp["pos_r"] = np.where(df1 > target.shift(), 1, 0)  # 只做多，不做空
        df_tmp['return_base'] = np.log(target / target.shift(1))  # 转化为对数后计算收益率
        df_tmp['return_s'] = df_tmp['pos_s'] * df_tmp['return_base']
        df_tmp['return_r'] = df_tmp['pos_r'] * df_tmp['return_base']
        df_tmp.fillna(0, inplace=True)
        # 累计收益
        ret_base, ret_model, ret_model_raw = np.exp(df_tmp[['return_base', 'return_s', 'return_r']].sum())
        # 信息输出
        logger.info("{model}模型基于MAE评价的结果是{mae}", model=model, mae=round(mae, 4))
        logger.info("{model}模型基于RMSE评价的结果是{rmse}", model=model, rmse=round(rmse, 4))
        logger.info("{model}模型基于R2评价的结果是{r2}", model=model, r2=round(r2, 4))
        logger.info("{model}模型基于MAPE评价的结果是{mape}", model=model, mape=round(mape, 4))
        logger.info("{model}模型的准确度，精确度，F1值分别是{accu}, {prec}, {f1}"
                    , model=model, accu=accu, prec=prec, f1=f1)
        logger.info("基础收益率是{ret_base}, {model}模型收益率(有做空)是{ret}, 无做空为{ret_r}"
                    , ret_base=round(ret_base, 4), model=model, ret=round(ret_model, 4), ret_r=round(ret_model_raw, 4))
        # 计算阶段累计收益详情
        df_tmp['基础收益率'] = np.exp(df_tmp['return_base'].cumsum())
        df_tmp['模型收益率'] = np.exp(df_tmp['return_s'].cumsum())
        df_tmp['模型收益率（无做空）'] = np.exp(df_tmp['return_r'].cumsum())
        df_cum = df_tmp[["基础收益率", "模型收益率", "模型收益率（无做空）"]]  # 累计收益
        if ret_info:
            df_cum.plot(figsize=(9, 4), style=['k-', 'r-', 'g:'])
            plt.title("模型收益率对比图")
            plt.show()
        return df_tmp

    def report(self, lstm_out, gru_out, tcn_out, target, args):
        """
        记录IMFS预测数据，并进行评估
        :param lstm_out: 多个IMFS的LSTM预测值
        :param gru_out: 多个IMFS的GRU模型预测值
        :param tcn_out: 多个IMFS的TCN模型预测值
        :param target: 待预测训练集的真实值y
        :param args: 参数
        :return: none
        """
        # 将多个IMFS加总作为最终预测值
        lstm_pred, gru_pred, tcn_pred = sum(lstm_out), sum(gru_out), sum(tcn_out)
        logger.info("LSTM模型预测值: {total}, GRU模型预测值: {gru}, TCN模型预测值: {tcn}",
                    total=round(lstm_pred, 4), gru=round(gru_pred, 4), tcn=round(tcn_pred, 4))
        # 将结果写入csv文件
        if type(args.file_name) == str:  # 续传
            path = "./output/" + args.file_name
        else:
            path = ("./output/output_" + str(int(args.file_name)) + ".csv")
        with open(path, 'a', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow((lstm_pred, gru_pred, tcn_pred, target))
        # 训练快完成时，读取历史数据并评估
        df_hist = pd.read_csv(path)
        if df_hist.shape[0] > 98:
            # 输出模型预测误差
            self.eval_all(df_hist["lstm_pred"], df_hist["target"], model="LSTM")
            self.eval_all(df_hist["gru_pred"], df_hist["target"], model="gru")
            self.eval_all(df_hist["tcn_pred"], df_hist["target"], model="tcn")
        return None


class TrainingProc(DataProcess):

    def train_emd(self, args, emd_data, clean_data, j):
        """
        单次模型训练框架。输入所有解释变量与被解释变量，通过参数j实现并行（多进程）运算
        :param args: 预设超参数
        :param emd_data: 所有被解释变量
        :param clean_data: 所有解释变量
        :param j: 控制并行运算参数
        :return: 模型的预测值
        """
        # 对EMD分解后的多个数据分别进行预测,后合并
        emd_j = emd_data.iloc[:, j:j + 1]
        # 标准化EMD数据
        scaler_emd = MinMaxScaler()
        emd_j_sc = scaler_emd.fit_transform(emd_j)
        emd_j_sc = pd.DataFrame(emd_j_sc)
        data = pd.concat([clean_data, emd_j_sc], axis=1)
        # 重塑数据
        seq_len = args.model_training_data_len  # 每个样本的时间长度
        train_x, train_y, test_x = self.convert_sup(data, seq_len, args.batch_size)
        # 分别调用模型
        # GRU
        pred_gru_sc = ModelTraining.train_gru(train_x, train_y, test_x, args)
        pred_gru = float(scaler_emd.inverse_transform(pred_gru_sc))
        # pred_gru = 1
        # TCN
        pred_tcn_sc = ModelTraining.train_tcn(train_x, train_y, test_x, args)
        pred_tcn = float(scaler_emd.inverse_transform(pred_tcn_sc))
        # pred_tcn = 1
        # LSTM
        pred_lstm_sc = ModelTraining.train_lstm(train_x, train_y, test_x, args)
        pred_lstm = float(scaler_emd.inverse_transform(pred_lstm_sc))
        # pred_lstm = 1
        return pred_lstm, pred_gru, pred_tcn

    def start_training(self, args):
        """
        模型训练启动函数。
        包括：数据加载并进行EMD处理，开始训练，导出结果
        :param args: 设置超参数
        :return: None
        """
        # 记录数据
        self.initial_file(args)
        # 加载数据，区分是否跨市场的数据情况
        if args.with_mkt:
            raw_data = pd.read_csv(r"./raw_data/SH300_with_HSI_SPX.csv")
        else:
            raw_data = self.prep_basic_data(code=args.data_type)
        # 断点处理，支持从指定位置开始训练
        start = args.starting_point
        end = start + 100 - args.starting_point
        for i in tqdm(range(start, end)):
            logger.info("program 第{i}次训练开始", i=i)
            train_data, target = self.get_window_data(raw_data, i, interval=args.training_interval)
            logger.info("program 模型的真实值为{target}", target=target)
            # EMD转换
            emd_data = self.convert_imfs(train_data)
            # 标准化原始数据
            train_data_new = train_data.drop(columns=["Date", "IndexCode"])
            scaler = MinMaxScaler()
            clean_data = scaler.fit_transform(train_data_new)
            clean_data = pd.DataFrame(clean_data, columns=train_data_new.columns)
            # 存放输出数据
            lstm_out, gru_out, tcn_out = [], [], []
            # 并行运算，根据分解的IMF数量设定进程数
            j_list = [j for j in range(emd_data.shape[1])]
            para = partial(self.train_emd, args, emd_data, clean_data)
            with Pool() as p:
                results = p.map(para, j_list)
                for pred_model, pred_gru, pred_tcn in results:
                    lstm_out.append(pred_model)
                    gru_out.append(pred_gru)
                    tcn_out.append(pred_tcn)
            self.report(lstm_out, gru_out, tcn_out, target, args)


class DataEval(DataProcess):

    def plot_emd_sample(self):
        """
        以历史数据为例，画EMD分解图
        :return: 画图的原始数据
        """
        raw_data = pd.read_csv(r"./raw_data/SH300_with_HSI_SPX.csv")
        df, target = self.get_window_data(raw_data, 0, interval=2792)
        s = df[["CloseIndex", "Date"]]
        emd_data = self.convert_imfs(df)
        df_plot = pd.concat([s, emd_data], axis=1)
        df_plot.rename(columns={"emd_7": "res"}, inplace=True)
        df_plot.set_index("Date", inplace=True)
        # united emd plot
        df_plot.plot(figsize=(9, 6))
        plt.ylabel("Index")
        plt.show()
        # divided emd plot
        fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(6, 12))
        cols = df_plot.columns
        for i in range(7):
            df_plot.iloc[:, i+1].plot(ax=axes[i])
            axes[i].set_title(cols[i+1])
            axes[i].set_ylabel('Index')
        plt.show()
        return df_plot

    def combo_model(self, file_path, a0=0.99):
        """
        通过强化学习框架，组合各模型的结果，观察是否由于基础模型
        :param a0: 超参数a0
        :param file_path: 保存数据的文件路径
        :return: None
        """

        def update_weight(error_list, weight_list, num_model, a):
            new_weight_list = []
            for i in range(len(weight_list)):
                new_weight = a * weight_list[i] + ((1 - a) / (num_model - 1)) * (1 - error_list[i])
                new_weight_list.append(new_weight)
            return new_weight_list

        def eval_3(df_input, a=a0, num_model=3):
            df = df_input.copy(deep=True)
            df["gru_error"] = abs(df["gru_pred"] - df["target"])
            df["tcn_error"] = abs(df["tcn_pred"] - df["target"])
            df["lstm_error"] = abs(df["lstm_pred"] - df["target"])
            df["all_error"] = df["gru_error"] + df["tcn_error"] + df["lstm_error"]
            df["gru_rate"] = df["gru_error"] / df["all_error"]
            df["tcn_rate"] = df["tcn_error"] / df["all_error"]
            df["lstm_rate"] = df["lstm_error"] / df["all_error"]
            all_weight = [[1 / num_model] * num_model]
            for i in range(df.shape[0] - 1):  # 最后一天不需要计算了
                init_weight = all_weight[-1]
                step = df.loc[i, ["gru_rate", "tcn_rate", "lstm_rate"]].to_list()
                step_weight = update_weight(step, init_weight, num_model=num_model, a=a)
                all_weight.append(step_weight)
            df_weight = pd.DataFrame(all_weight, columns=["gru_w", "tcn_w", "lstm_w"])
            # 计算综合后的结果
            df["prediction"] = df["gru_pred"] * df_weight["gru_w"] + df["tcn_pred"] * df_weight["tcn_w"] \
                               + df["lstm_pred"] * df_weight["lstm_w"]
            return df["prediction"]

        def eval_2(df_input, model, a=a0):
            df = df_input.copy(deep=True)
            if "lstm" not in model:
                names = ["gru_pred", "tcn_pred"]
            elif "gru" not in model:
                names = ["lstm_pred", "tcn_pred"]
            elif "tcn" not in model:
                names = ["lstm_pred", "gru_pred"]
            else:
                raise NameError
            df["0_error"] = abs(df[names[0]] - df["target"])
            df["1_error"] = abs(df[names[1]] - df["target"])
            df["all_error"] = df["0_error"] + df["1_error"]
            df["0_rate"] = df["0_error"] / df["all_error"]
            df["1_rate"] = df["1_error"] / df["all_error"]
            all_weight = [[0.5, 0.5]]
            for i in range(df.shape[0] - 1):  # 最后一天不需要计算了
                init_weight = all_weight[-1]
                step = df.loc[i, ["0_rate", "1_rate"]].to_list()
                step_weight = update_weight(step, init_weight, num_model=2, a=a)
                all_weight.append(step_weight)
            df_weight = pd.DataFrame(all_weight, columns=["0_w", "1_w"])
            # 计算综合后的结果
            df["prediction"] = df[names[0]] * df_weight["0_w"] + df[names[1]] * df_weight["1_w"]
            return df["prediction"]

        df_raw = pd.read_csv(file_path)
        rest_3 = eval_3(df_raw)
        # gru_tcn = eval_2(df_raw, model=["gru", "tcn"])
        # lstm_tcn = eval_2(df_raw, model=["lstm", "tcn"])
        # gru_lstm = eval_2(df_raw, model=["lstm", "gru"])
        # 评估
        self.eval_all(df_raw["gru_pred"], df_raw["target"], model="GRU", disp_cm=False, ret_info=False)
        self.eval_all(df_raw["tcn_pred"], df_raw["target"], model="TCN", disp_cm=False, ret_info=False)
        self.eval_all(df_raw["lstm_pred"], df_raw["target"], model="LSTM", disp_cm=False, ret_info=False)
        # self.eval_all(gru_tcn, df_raw["target"], model="gru_tcn", disp_cm=False, ret_info=True)
        # self.eval_all(lstm_tcn, df_raw["target"], model="lstm_tcn", disp_cm=False, ret_info=True)
        # self.eval_all(gru_lstm, df_raw["target"], model="gru_lstm", disp_cm=False, ret_info=True)
        df_3 = self.eval_all(rest_3, df_raw["target"], model="all three model", disp_cm=False, ret_info=True)
        # 画图，拟合情况
        # df_raw["gru_tcn"] = gru_tcn
        # df_raw["lstm_tcn"] = lstm_tcn
        # df_raw["gru_lstm"] = gru_lstm
        df_raw["all_pred"] = rest_3
        df_out = df_raw[["target", "gru_pred", "tcn_pred", "lstm_pred", "all_pred"]]
        df_out.rename(columns={"target": "目标值",
                               "gru_pred": "GRU 预测值",
                               "tcn_pred": "TCN 预测值",
                               "lstm_pred": "LSTM 预测值",
                               "all_pred": "HML 预测值",
                               }, inplace=True)
        df_out.plot(figsize=(9, 4), style=['k-', 'c:', 'm:', 'y:', 'g-'])
        plt.title("多模型预测值对比")
        plt.show()
        return df_3

    def find_every_best(self, path):
        """
        输入多次训练的预测值，逐日找出最佳预测。
        :return:最优预测值
        """

        def func1(x):
            y = x if abs(x) < 30 else np.sign(x) * 30
            return y

        df1 = pd.read_csv(path)
        cols = df1.columns
        df1["lstm_a"] = df1["lstm_pred"] - df1["target"].shift()
        df1["gru_a"] = df1["gru_pred"] - df1["target"].shift()
        df1["tcn_a"] = df1["tcn_pred"] - df1["target"].shift()
        df1["lstm_aa"] = df1["lstm_a"].map(func1)
        df1["gru_aa"] = df1["gru_a"].map(func1)
        df1["tcn_aa"] = df1["tcn_a"].map(func1)
        df1["lstm"] = df1["lstm_aa"] + df1["target"].shift()
        df1["gru"] = df1["gru_aa"] + df1["target"].shift()
        df1["tcn"] = df1["tcn_aa"] + df1["target"].shift()
        df1.loc[0, "lstm"] = df1.loc[0, "lstm_pred"]
        df1.loc[0, "gru"] = df1.loc[0, "gru_pred"]
        df1.loc[0, "tcn"] = df1.loc[0, "tcn_pred"]
        df_out = df1[["lstm", "gru", "tcn", "target"]]
        df_out.columns = cols
        new_path = path[:-4] + "_new.csv"
        df_out.to_csv(new_path, index=False)
        self.combo_model(new_path)
        pass

    @staticmethod
    def plot_radar():
        def num_data():
            numdata = [
                ['MAE', 'RMSE', 'R2', 'MAPE'],
                ('基于跨市场牛市回调', [
                    [0.38, 0.37, 0.94, 0.40],
                    [0.10, 0.19, 0.91, 0.10],
                    [0.14, 0.24, 0.94, 0.15],
                    [0.79, 0.73, 1.00, 0.85]]),
                ('基于单市场牛市回调', [
                    [0.49, 0.48, 0.95, 0.52],
                    [0.01, 0.08, 0.90, 0.00],
                    [0.00, 0.00, 0.89, 0.02],
                    [0.65, 0.64, 0.99, 0.71]]),
                ('基于跨市场熊市反转', [
                    [0.32, 0.39, 0.00, 0.23],
                    [0.26, 0.40, 0.09, 0.15],
                    [0.56, 0.63, 0.38, 0.50],
                    [0.96, 0.99, 0.52, 0.96]]),
                ('基于单市场熊市反转', [
                    [0.67, 0.73, 0.25, 0.63],
                    [0.72, 0.79, 0.37, 0.69],
                    [0.66, 0.66, 0.32, 0.62],
                    [1.00, 1.00, 0.52, 1.00]])
            ]
            return numdata

        def rate_data():
            ratedata = [
                ['准确率', '精度', 'F1值'],
                ('基于跨市场牛市回调', [
                    [0.6465, 0.6792, 0.6729],
                    [0.6364, 0.8000, 0.5714],
                    [0.6869, 0.7255, 0.7048],
                    [0.7071, 0.8049, 0.6947]]),
                ('基于单市场牛市回调', [
                    [0.6667, 0.6981, 0.6916],
                    [0.6263, 0.7931, 0.5542],
                    [0.6768, 0.7200, 0.6923],
                    [0.6970, 0.7629, 0.7000]]),
                ('基于跨市场熊市反转', [
                    [0.5556, 0.5309, 0.5417],
                    [0.6263, 0.6042, 0.6105],
                    [0.5758, 0.5410, 0.6111],
                    [0.5960, 0.5614, 0.6154]]),
                ('基于单市场熊市反转', [
                    [0.5657, 0.5909, 0.3768],
                    [0.5253, 0.5000, 0.4719],
                    [0.5455, 0.5185, 0.5545],
                    [0.5758, 0.5676, 0.5000]])
            ]
            return ratedata

        def radar_factory(num_vars, frame='circle'):
            """
            Create a radar chart with `num_vars` axes.

            This function creates a RadarAxes projection and registers it.

            Parameters
            ----------
            num_vars : int
                Number of variables for radar chart.
            frame : {'circle', 'polygon'}
                Shape of frame surrounding axes.

            """
            # calculate evenly-spaced axis angles
            theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

            class RadarTransform(PolarAxes.PolarTransform):

                def transform_path_non_affine(self, path):
                    # Paths with non-unit interpolation steps correspond to gridlines,
                    # in which case we force interpolation (to defeat PolarTransform's
                    # autoconversion to circular arcs).
                    if path._interpolation_steps > 1:
                        path = path.interpolated(num_vars)
                    return Path(self.transform(path.vertices), path.codes)

            class RadarAxes(PolarAxes):

                name = 'radar'
                # use 1 line segment to connect specified points
                RESOLUTION = 1
                PolarTransform = RadarTransform

                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    # rotate plot such that the first axis is at the top
                    self.set_theta_zero_location('N')

                def fill(self, *args, closed=True, **kwargs):
                    """Override fill so that line is closed by default"""
                    return super().fill(closed=closed, *args, **kwargs)

                def plot(self, *args, **kwargs):
                    """Override plot so that line is closed by default"""
                    lines = super().plot(*args, **kwargs)
                    for line in lines:
                        self._close_line(line)

                def _close_line(self, line):
                    x, y = line.get_data()
                    # FIXME: markers at x[0], y[0] get doubled-up
                    if x[0] != x[-1]:
                        x = np.append(x, x[0])
                        y = np.append(y, y[0])
                        line.set_data(x, y)

                def set_varlabels(self, labels):
                    self.set_thetagrids(np.degrees(theta), labels)

                def _gen_axes_patch(self):
                    # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                    # in axes coordinates.
                    if frame == 'circle':
                        return Circle((0.5, 0.5), 0.5)
                    elif frame == 'polygon':
                        return RegularPolygon((0.5, 0.5), num_vars,
                                              radius=.5, edgecolor="b")
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)

                def _gen_axes_spines(self):
                    if frame == 'circle':
                        return super()._gen_axes_spines()
                    elif frame == 'polygon':
                        # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                        spine = Spine(axes=self,
                                      spine_type='circle',
                                      path=Path.unit_regular_polygon(num_vars))
                        # unit_regular_polygon gives a polygon of radius 1 centered at
                        # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                        # 0.5) in axes coordinates.
                        spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                            + self.transAxes)
                        return {'polar': spine}
                    else:
                        raise ValueError("Unknown value for 'frame': %s" % frame)

            register_projection(RadarAxes)
            return theta

        kind = "num"
        if kind == "num":
            N = 4
            theta = radar_factory(N, frame='polygon')
            data = num_data()

            spoke_labels = data.pop(0)

            fig, axs = plt.subplots(figsize=(8, 8), nrows=2, ncols=2,
                                    subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(wspace=0.25, hspace=0.20, top=1.20, bottom=0.05)

            colors = ['c', 'g', 'b', 'r']
            # Plot the four cases from the example data on separate axes
            for ax, (title, case_data) in zip(axs.flat, data):
                ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                             horizontalalignment='center', verticalalignment='center')
                for d, color in zip(case_data, colors):
                    ax.plot(theta, d, color=color)
                    ax.set_rmax(1.0)
                    ax.set_rmin(0)
                #             ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
                ax.set_varlabels(spoke_labels)

            # add legend relative to top-left plot
            labels = ('LSTM', 'GRU', 'TCN', 'HML')
            legend = axs[0, 0].legend(labels, loc=(1, 0),
                                      labelspacing=0.2, fontsize='x-large', )

            fig.text(0.5, 0.965, '绝对值指标族对比雷达图',
                     horizontalalignment='center', color='black', weight='bold',
                     size='xx-large')

            plt.show()

        elif kind == 'rate':
            N = 3
            theta = radar_factory(N, frame='polygon')
            data = rate_data()

            spoke_labels = data.pop(0)

            fig, axs = plt.subplots(figsize=(7, 7), nrows=2, ncols=2,
                                    subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(wspace=0.25, hspace=0.20, top=1.20, bottom=0.05)

            colors = ['c', 'g', 'b', 'r']
            # Plot the four cases from the example data on separate axes
            for ax, (title, case_data) in zip(axs.flat, data):
                ax.set_rgrids([0.4, 0.6, 0.8])
                ax.set_title(title, weight='bold', size='large', position=(0.5, 1.1),
                             horizontalalignment='center', verticalalignment='center')
                for d, color in zip(case_data, colors):
                    ax.plot(theta, d, color=color)
                    ax.set_rmax(0.81)
                    ax.set_rmin(0.4)
                #             ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
                ax.set_varlabels(spoke_labels)

            # add legend relative to top-left plot
            labels = ('LSTM', 'GRU', 'TCN', 'HML')
            legend = axs[0, 0].legend(labels, loc=(1, 0.5),
                                      labelspacing=0.2, fontsize='x-large', )

            fig.text(0.5, 0.965, '精度值指标族对比雷达图',
                     horizontalalignment='center', color='black', weight='bold',
                     size='xx-large')

            plt.show()

    @staticmethod
    def plot_bar():
        labels = ['LSTM', 'GRU', 'TCN', 'HML']
        mkt_bull = [0.4892, 0.672, 0.6439, 0.854]
        nomkt_bull = [0.5376, 0.5322, 0.7266, 0.8114]
        mkt_bear = [0.3695, 0.1986, 0.3364, 0.3839]
        nomkt_bear = [0.3633, 0.2342, 0.2701, 0.3801]

        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        # 子图
        plt.figure(figsize=(6, 7))
        plt.subplot(211)
        plt.bar(x - 0.5 * width, nomkt_bull, width, label='不基于跨市场数据')
        plt.bar(x + 0.5 * width, mkt_bull, width, label='基于跨市场数据')
        plt.ylabel('超额收益率')
        plt.title("牛市回调市场的超额收益率对比")
        plt.xticks(x, labels)
        plt.xticks(rotation=-15)
        plt.ylim(top=0.9)
        plt.legend()

        plt.subplot(212)
        plt.bar(x - 0.5 * width, nomkt_bear, width, label='不基于跨市场数据')
        plt.bar(x + 0.5 * width, mkt_bear, width, label='基于跨市场数据')
        plt.ylabel('超额收益率')
        plt.title("熊市反转市场的超额收益率对比")
        plt.xticks(x, labels)
        plt.xticks(rotation=-15)
        plt.ylim(top=0.9)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Price Prediction EMD')
    # data
    parser.add_argument('--data_type', default="000300", type=str)  # 指定沪深300数据
    parser.add_argument('--training_interval', default=2792, type=int)  # 指定预测数据集：2792熊市，2628牛市
    parser.add_argument('--starting_point', default=0, type=int)  # 运行程序中的断点
    parser.add_argument('--file_name', default=time.time(), type=int)  # 用于生成唯一的文件名
    parser.add_argument('--model_training_data_len', default=25, type=int)  # 用于规定单个样本的数据包含的时间长度
    parser.add_argument('--with_mkt', default=1, type=int)  # 指定是否包含跨市场数据
    # LSTM
    parser.add_argument('--lstm_cell_1', default=64, type=int)  # 第一层cell数
    parser.add_argument('--lstm_cell_2', default=64, type=int)  # 第二层cell数
    # GRU
    parser.add_argument('--gru_cell_1', default=64, type=int)
    parser.add_argument('--gru_cell_2', default=64, type=int)
    # LSTM, GRU通用
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epoch', default=150, type=int)
    # TCN
    parser.add_argument('--tcn_epoch', default=200, type=int)
    parser.add_argument('--tcn_nb_filters', default=64, type=int)  # 类似于lstm中的cell
    parser.add_argument('--tcn_batch_size', default=256, type=int)
    arg = parser.parse_args()
    # # 训练模型
    # tp = TrainingProc()
    # tp.start_training(args=arg)
    # # 评估模型
    # de = DataEval()
    # data_path = "./evaluate/down/with_mkt/train3.csv"
    # df = de.combo_model(data_path)
    # DataEval.plot_radar()
    de = DataEval()
    emd = de.plot_emd_sample()
