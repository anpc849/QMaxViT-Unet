import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if gt.sum() == 0 and pred.sum() == 0:
        return np.nan, np.nan
    elif gt.sum() == 0 and pred.sum() > 0:
        return 0, 0
    elif gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        if pred.sum() == 0:
            hd95 = np.nan
        else:
            hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95


@torch.no_grad()
def test_single_volume_for_training(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction_1 = np.zeros_like(label)
        prediction_2 = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P1,P2,_ = net(input)

                iout_soft1 = torch.softmax(P1, dim=1)
                #iout_soft2 = torch.softmax(P2, dim=1)
                out_1 = torch.argmax(iout_soft1, dim=1).squeeze(0)
                #out_2 = torch.argmax((iout_soft2+iout_soft1), dim=1).squeeze(0)
                #out_2 = out_2.cpu().detach().numpy()
                out_1 = out_1.cpu().detach().numpy()
                
                pred_1 = zoom(
                    out_1, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction_1[ind] = pred_1
                
                # pred_2 = zoom(
                #     out_2, (x / patch_size[0], y / patch_size[1]), order=0)
                # prediction_2[ind] = pred_2
                
    else:
        input = torch.from_numpy(image).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            # val_outputs = 0.0
            # for idx in range(len(P)):
            #     val_outputs += P[idx]

            iout_soft = torch.softmax(P, dim=1)
            out = torch.argmax(iout_soft, dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    metric_list_one = []
    metric_list_two = []
    for i in range(1, classes):
        metric_list_one.append(calculate_metric_percase(
            prediction_1 == i, label == i))
        # metric_list_two.append(calculate_metric_percase(
        #     prediction_2 == i, label == i))
    return metric_list_one