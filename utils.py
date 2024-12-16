import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import torch
import torch.nn.functional as F

#logistic_func：该函数使用具有参数bayta1、bayta2、bayta3和bayta4的 logistic 函数来计算预测值。它计算 logistic 部分并返回预测值 yhat
def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

#该函数使用scipy.optimize.curve_fit从给定的数据点(y_label和y_output)中拟合 logistic 曲线。
# 它初始化参数beta并使用曲线拟合找到这些参数的最佳值。然后，它基于拟合的参数计算 logistic 预测并返回它们
def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE

#该函数评估了没有 logistic 曲线拟合的模型的性能。它直接计算与performance_fit相同的评估指标，但不执行 logistic 曲线拟合
def performance_no_fit(y_label, y_output):
    PLCC = stats.pearsonr(y_output, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output-y_label) ** 2).mean())
################################################################y_label-y_label
    return PLCC, SRCC, KRCC, RMSE


#这是一个自定义的 PyTorch 损失函数类，称为 L1RankLoss。它组合了 L1 损失和排名（Rank）损失。以下是这个损失函数的主要特点和功能
class L1RankLoss(torch.nn.Module):
    """
    L1 loss + Rank loss
    """
    # l1_w：L1
    # 损失的权重。
    # rank_w：排名损失的权重。
    # hard_thred：用于确定是否为
    # "hard"
    # 样本的阈值。
    # use_margin：一个布尔值，指示是否使用
    # margin。
    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)
        self.rank_w = kwargs.get("rank_w", 1)
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)

#preds（模型的预测值）和 gts（真实值）
    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss    模型预测值和真实值之间的平均绝对误差，并乘以 l1_w 权重
        preds = preds.view(gts.size())
###################
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        return loss_total