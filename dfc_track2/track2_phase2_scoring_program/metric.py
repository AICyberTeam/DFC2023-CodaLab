import os
import math
import json
import numpy as np
from sys import argv
from coco import CocoDataset
from libtiff import TIFF, TIFF3D
from osgeo import gdal

class Result(object):
    def __init__(self):
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel = 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def set_to_worst(self):
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel = np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, mse, rmse, mae, absrel, delta1, delta2, delta3):
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel = absrel
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3

    def evaluate(self, output, target):
        output[output == 0] = 0.00001
        output[output < 0] = 999
        target[target <= 0] = 0.00001
        valid_mask = ((target>0) + (output>0)) > 0

        output = output[valid_mask]
        target = target[valid_mask]
        abs_diff = np.abs(output - target)

        self.mse = np.mean(abs_diff ** 2)
        self.rmse = np.sqrt((abs_diff ** 2).mean())
        self.mae = np.mean(abs_diff)

        self.absrel = float((abs_diff / target).mean())

        maxRatio = np.maximum(output / target, target / output)
        self.delta1 = (maxRatio < 1.25).mean()
        self.delta2 = (maxRatio < 1.25 ** 2).mean()
        self.delta3 = (maxRatio < 1.25 ** 3).mean()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel = 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, result, n=1):
        self.count += n
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3

    def average(self):
        avg = Result()
        avg.update(
            self.sum_mae / self.count, self.sum_mse / self.count, self.sum_rmse / self.count,
            self.sum_absrel / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)
        return avg


def evaluate_track_2(gt_file, results_file):
    CLASSES = ('building',)
    coco_mmdet = CocoDataset(gt_file)
    metric = ['segm']
    metrics = metric if isinstance(metric, list) else [metric]
    allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

    coco_gt = coco_mmdet.coco
    coco_mmdet.cat_ids = coco_gt.get_cat_ids(cat_names=CLASSES)
    result_files = dict()
    result_files['segm'] = results_file
    eval_results, results_classwise = coco_mmdet.evaluate_det_segm(None, result_files, coco_gt,
                                                metrics, None, True)
    return eval_results, results_classwise

if __name__ == '__main__':

    '''Result Path'''
    '''
    ---ref
       ---gt.json
       ---gt_dsm
          ---1.tif
          ---2.tif
    ---res
       ---Result.json
       ---DSM_Result
          ---1.tif
          ---2.tif
          ...
    '''


    input_dir = argv[1]
    gt_path = os.path.join(input_dir, 'ref')
    pred_path = os.path.join(input_dir, 'res')

    seg_gt_path = os.path.join(gt_path, 'buildings_only_test.json')
    dsm_gt_path = os.path.join(gt_path, 'height')
    dsm_gt_list = os.listdir(dsm_gt_path)

    seg_pred_path = os.path.join(pred_path, 'seg_results.json')
    dsm_pred_path = os.path.join(pred_path, 'height')
    dsm_pred_list = os.listdir(dsm_pred_path)

    output_dir = argv[2]
    output_file = os.path.join(output_dir, 'scores.txt')

    '''Seg Metric'''
    results_summary, results_classwise = evaluate_track_2(seg_gt_path, seg_pred_path)
    print(results_summary, results_classwise)

    '''DSM Metric'''
    average_meter = AverageMeter()
    for dsm_pred_i in dsm_pred_list:
        dsm_pred_path_i = os.path.join(dsm_pred_path, dsm_pred_i)
        dsm_gt_path_i = os.path.join(dsm_gt_path, dsm_pred_i)
        print(dsm_pred_path_i)
        print(dsm_gt_path_i)

        dsm_pred_tif = gdal.Open(dsm_pred_path_i)
        dsm_pred_array = dsm_pred_tif.ReadAsArray()
        dsm_gt_tif = gdal.Open(dsm_gt_path_i)
        dsm_gt_array = dsm_gt_tif.ReadAsArray()

        dsm_pred_array = np.squeeze(dsm_pred_array)
        dsm_gt_array = np.asarray(dsm_gt_array, np.float32)
        dsm_gt_array = np.array(dsm_gt_array)
        dsm_gt_array = np.squeeze(dsm_gt_array)

        result = Result()
        result.evaluate(dsm_pred_array, dsm_gt_array)
        average_meter.update(result)
    avg = average_meter.average()

    '''DSM Metric'''
    MAE, RMSE, Delta1, Delta2, Delta3 = avg.mae, avg.rmse, avg.delta1, avg.delta2, avg.delta3
    '''Seg Metric'''
    mAP, mAP50 = results_summary['segm_mAP'], results_summary['segm_mAP_50']

    final_metric = (mAP50 + Delta1)/2.0
    with open(output_file, 'w') as out:
        out.write('mAP' + ':' + str(mAP)[:5] + '\n')
        out.write('AP_50' + ':' + str(mAP50)[:5] + '\n')
        out.write('Delta_1' + ':' + str(Delta1)[:6] + '\n')
        out.write('Delta_2' + ':' + str(Delta2)[:6] + '\n')
        out.write('Delta_3' + ':' + str(Delta3)[:6] + '\n')
        out.write('Score' + ':' + str(final_metric)[:6] + '\n')
        out.close()
