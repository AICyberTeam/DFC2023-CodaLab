from coco import CocoDataset


# CLASSES = (
#     "flat_roof",
#     "gable_roof",
#     "gambrel_roof",
#     "row_roof",
#     "multiple_eave_roof",
#     "hipped_roof_v1",
#     "hipped_roof_v2",
#     "mansard_roof",
#     "pinnacle_roof",
#     "arched_roof",
#     "dome",
#     "other"
# )
# CLASSES = (
#     'flat_roof',
#     'flat_roof_HVAC',
#     'flat_roof_complex',
#     'shed_roof',
#     'gable_roof',
#     'gable_roof_asymm',
#     'gambrel_roof',
#     'row_roof_shed',
#     'row_roof_gable',
#     'row_roof_arched',
#     'multiple_eave_roof',
#     'hipped_roof_v1',
#     'hipped_roof_v2',
#     'half_hipped_roof',
#     'mansard_roof',
#     'pinnacle_roof',
#     'arched_roof',
#     'dome',
#     'freeshape',
#     'other'
# )

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

    input_dir = argv[1]
    output_dir = argv[2]
    gt_file_fine = os.path.join(input_dir, 'ref', 'buildings_test.json')
    segm_json_file = os.path.join(input_dir, 'res', 'results.segm.json')

    # gt_file = '/workspace/dataset/UBC_v2/demo_develop/track2/crop_512_RGB/annotations/buildings_test.json'
    # segm_json_file = '/workspace/dataset/UBC_v2/demo_develop/track2/demo_buildings_output/results.segm.json'
    results_summary, results_classwise =  evaluate_track_2(gt_file, segm_json_file)
    print(results_summary, results_classwise)