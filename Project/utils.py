import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    # Specify: criterion_classes_params = {'input': 'pred_classes', 'target': 'data_target'}
    def forward(self, input, target):
        pred_class0, pred_class1 = input[:, 0], input[:, 1]
        pred_target = (pred_class0.argmax(1) <= pred_class1.argmax(1)).float()

        loss = torch.mean((pred_target - target).pow(2))  # mse

        return loss.clone().detach().requires_grad_(True)


def _test():
    x0 = torch.tensor([
        [0.1135, 0.0899, 0.0696, 0.0920, 0.1168, 0.1083, 0.0842, 0.1261, 0.1029, 0.0966],
        [0.1033, 0.1058, 0.0679, 0.0930, 0.1008, 0.0924, 0.0908, 0.1022, 0.1358, 0.1080],
        [0.1166, 0.0951, 0.0770, 0.0741, 0.1057, 0.0949, 0.1055, 0.1110, 0.1184, 0.1018],
        [0.1300, 0.1031, 0.0715, 0.0904, 0.0918, 0.0859, 0.0932, 0.1090, 0.1244, 0.1006],
        [0.1139, 0.0931, 0.0776, 0.0737, 0.1254, 0.1021, 0.0932, 0.1370, 0.0975, 0.0866]
    ])
    x1 = torch.tensor([
        [0.1135, 0.0852, 0.0849, 0.0981, 0.1102, 0.0976, 0.0927, 0.0986, 0.1187, 0.1006],
        [0.1176, 0.0923, 0.0702, 0.0853, 0.1156, 0.1066, 0.0919, 0.1229, 0.1048, 0.0927],
        [0.1103, 0.0957, 0.0739, 0.0915, 0.0986, 0.0978, 0.0893, 0.1236, 0.1118, 0.1075],
        [0.1087, 0.0891, 0.0697, 0.0977, 0.1007, 0.1030, 0.1066, 0.1059, 0.1243, 0.0943],
        [0.1047, 0.0854, 0.0812, 0.0968, 0.1125, 0.0992, 0.0858, 0.1177, 0.1337, 0.0830]
    ])
    # x0 = torch.rand((5, 10))
    # x1 = torch.rand((5, 10))
    xs = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1)), dim=1)  # (5, 2, 10)
    y = torch.tensor([1, 0, 0, 1, 1])

    loss_func = ContrastiveLoss()
    print("loss:", loss_func(xs, y))


if __name__ == "__main__":
    _test()
