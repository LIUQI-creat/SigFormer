import torch
import torch.nn.functional as F
from torch import nn

class BoundaryRegressionLoss(nn.Module):
    """
    Args：
        preds: the output of model before softmax. (N*T)
        gts: Ground Truth. (N*T)
    Return:
        the value of BoundaryRegressionLoss
    """

    def __init__(self, threshold: float = 4, ignore_index: int = -1) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        
        gt = gt[1:] - gt[:-1]     # 后一个 - 前一个
        #print('before,',gt)  # [1,2,-1,-2,0]
        gt2 = torch.where(gt != 0,1, gt)  # 边界为1 ，其余为0

        pred = pred[1:]
        
        pred = pred.to(torch.float32)
        gt2 = gt2.to(torch.float32)
        
#         print('pred',pred[:2])
#         print('gt',gt2[:2])
        loss = self.bce(pred,gt2.unsqueeze(1))
        
        return loss

class OpWeightedKappaLoss(nn.Module):
    def __init__(self, num_classes=11, ignore_index=10, weightage="linear"):  # パラメータの設定など初期化処理を行う
        super().__init__()
        assert num_classes == 11 and ignore_index == 10
        num_classes = 10
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.weightage = weightage
        self.weights_0 = torch.tensor([0, 1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=torch.float32)
        weight_mat = []
        for i in range(10):
            weight_mat.append(torch.roll(self.weights_0, i, 0))

        if weightage == "linear":
            self.weight_mat = torch.stack(weight_mat)
        elif weightage == "quadratic":
            self.weight_mat = torch.stack(weight_mat) ** 2

        label_vec = torch.arange(num_classes, dtype=torch.float32)
        self.row_label_vec = torch.reshape(label_vec, (1, num_classes))
        self.col_label_vec = torch.reshape(label_vec, (num_classes, 1))

    def forward(self, y_pred, y_true, epsilon=1e-06):
        y_pred = y_pred[y_true != self.ignore_index]
        y_true = y_true[y_true != self.ignore_index]
        y_pred = y_pred[:, :10]
        y_true = torch.nn.functional.one_hot(y_true, self.num_classes)

        self.row_label_vec = self.row_label_vec.to(device=y_true.device)
        self.col_label_vec = self.col_label_vec.to(device=y_true.device)
        self.weight_mat = self.weight_mat.to(device=y_true.device)
        self.weights_0 = self.weights_0.to(device=y_true.device)

        y_true = y_true.to(dtype=self.col_label_vec.dtype)
        y_pred = y_pred.to(dtype=self.weight_mat.dtype)
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)
        batch_size = y_true.shape[0]
        cat_labels = torch.matmul(y_true, self.col_label_vec)
        cat_label_mat = torch.tile(cat_labels, [1, self.num_classes])
        row_label_mat = []
        for i in y_true.argmax(1).tolist():
            row_label_mat.append(torch.roll(self.weights_0 + i, i, 0))
        row_label_mat = torch.stack(row_label_mat)
        if self.weightage == "linear":
            weight = torch.abs(cat_label_mat - row_label_mat)
        elif self.weightage == "quadratic":
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, dim=0, keepdims=True)
        pred_dist = torch.sum(y_pred, dim=0, keepdims=True)
        w_pred_dist = torch.matmul(self.weight_mat.clone(), pred_dist.transpose(0, 1))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist))
        denominator /= torch.tensor(batch_size, dtype=denominator.dtype)
        if denominator == 0:
            loss = torch.zeros(1)[0]
        else:
            loss = numerator / denominator
        return torch.log(loss + epsilon)


def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()


class IBLoss(nn.Module):
    def __init__(self, num_classes=11, ignore_index=10, weight=None, alpha=10000.0):
        super(IBLoss, self).__init__()
        assert alpha > 0
        assert num_classes == 11 and ignore_index == 10
        num_classes = 10
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight

    def forward(self, input, target, features):
        grads = torch.sum(
            torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)), 1
        )  # N * 1
        ib = grads * features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        return ib_loss(F.cross_entropy(input, target, reduction="none", weight=self.weight), ib)


def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()


class IB_FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000.0, gamma=0.0):
        super(IB_FocalLoss, self).__init__()
        assert alpha > 0
        self.alpha = alpha
        self.epsilon = 0.001
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1)  # N * 1
        ib = grads * (features.reshape(-1))
        ib = self.alpha / (ib + self.epsilon)
        return ib_focal_loss(
            F.cross_entropy(input, target, reduction="none", weight=self.weight), ib, self.gamma
        )


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    # loss = (1 - p) ** gamma * input_values
    loss = (1 - p) ** gamma * input_values * 10
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, num_classes=11, ignore_index=10, weight=None, gamma=1.0, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        assert num_classes == 11 and ignore_index == 10
        num_classes = 10
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return focal_loss(
            F.cross_entropy(
                input,
                target,
                reduction="none",
                weight=self.weight,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            ),
            self.gamma,
        )
