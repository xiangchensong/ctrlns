import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from evaluation.isba_code import IoU, IoD, accuracy_wo_bg
from evaluation.tasl_code import compute_mof, compute_IoU_IoD
import pickle

def inference(fr_cls, raw_len):
    # fr_cls (t, cls)
    fr_cls = F.interpolate(fr_cls.transpose(0, 1).unsqueeze(0), raw_len, mode='linear').squeeze(0).transpose(0, 1)
    return torch.argmax(fr_cls, dim=1)


def test_all(epoch, model, test_loader, logger, device, args, bg_cls, fully_eva=True):
    model.eval()

    match, total = 0, 0
    pred_lst, gt_lst = [], []
    save_pred_list, save_gt_list, save_comb_list = [], [], []
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            feat = data['feat'].to(device)
            mask = data['mask'].to(device)
            b = feat.shape[0]

            out = model(feat, mask)
            for j in range(b):
                fr_cls = out['fr_cls'][j][mask[j]]
                pred = inference(fr_cls, data['raw_len'][j]).cpu()
                gt = data['raw_gt'][j]
                match += torch.sum(pred == gt)
                total += data['raw_len'][j]
                # save_pred_list.append(torch.unsqueeze(pred, dim=-1).numpy())
                # save_gt_list.append(torch.unsqueeze(gt, dim=-1).numpy())
                combine = np.concatenate([torch.unsqueeze(pred, dim=-1).numpy(), torch.unsqueeze(gt, dim=-1).numpy()], axis=-1)
                # print(torch.unsqueeze(pred, dim=-1).numpy().shape)
                save_comb_list.append(combine)
                pred_lst.append(pred.numpy())
                gt_lst.append(gt.numpy())
    
    # print(pred_lst.shape)
    # pred_lst = np.concatenate(pred_lst, axis=0)
    # exit()

    # save_pred = np.array(pred_lst)
    # save_gt = np.array(gt_lst)
    with open('pred.pkl', 'wb') as f:
        pickle.dump(save_comb_list, f)

    acc = compute_mof(gt_lst, pred_lst)
    if logger is not None:
        logger.add_scalar('acc', acc, epoch)
    ret = {'acc': acc}
    if args.dataset == 'hollywood' or args.dataset == 'crosstask' or args.dataset == 'breakfast':
        acc_bg = accuracy_wo_bg(pred_lst, gt_lst, bg_class=bg_cls)
        print('test acc-bg: {}'.format(acc_bg))
        if logger is not None:
            logger.add_scalar('acc-bg', acc_bg, epoch)
    if fully_eva:
        acc_bg = accuracy_wo_bg(pred_lst, gt_lst, bg_class=bg_cls)
        iou_isba, iod_isba = IoU(pred_lst, gt_lst), IoD(pred_lst, gt_lst)
        iou_tasl, _, iod_tasl, _ = compute_IoU_IoD(gt_lst, pred_lst)
        ret['acc-bg'] = acc_bg
        ret['iou_isba'] = iou_isba
        ret['iod_isba'] = iod_isba
        ret['iou_tasl'] = iou_tasl
        ret['iod_tasl'] = iod_tasl
        print("IOU:", iou_tasl, "IOD:", iod_tasl)
    return ret
