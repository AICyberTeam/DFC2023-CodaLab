import pycocotools
import libtiff

from coco import CocoDataset
from sys import argv
import os


CLASSES_track_1_fine = (
    'flat_roof',
    'gable_roof',
    'gambrel_roof',
    'row_roof',
    'multiple_eave_roof',
    'hipped_roof_v1',
    'hipped_roof_v2',
    'mansard_roof',
    'pyramid_roof',
    'arched_roof',
    'revolved',
    'other',
)

def evaluate_all(gt_file, results_file, class_tupple, coarse=False):
    CLASSES = class_tupple
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

def evaluate_track_1(gt_file_fine, results_file, output_file):
    results_summary_fine, results_classwise_fine = evaluate_all(gt_file_fine, results_file, CLASSES_track_1_fine)
    print(results_summary_fine, results_classwise_fine)
    with open(output_file, 'w') as f:
        f.write('mAP:{:.3f}\n'.format(results_summary_fine['segm_mAP']))
        f.write('AP_50:{:.3f}\n'.format(results_summary_fine['segm_mAP_50']))
        for t in results_classwise_fine:
            f.write('mAP_{}:{}\n'.format(t[0], t[1] if 'nan' not in t[1] else 0.0))

if __name__ == '__main__':
    input_dir = argv[1]
    output_dir = argv[2]
    gt_file_fine = os.path.join(input_dir, 'ref', 'roof_fine_val.json')
    segm_json_file = os.path.join(input_dir, 'res', 'results.json')
    output_file = os.path.join(output_dir, 'scores.txt')
    evaluate_track_1(gt_file_fine, segm_json_file, output_file)
    print("Done!")