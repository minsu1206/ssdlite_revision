import torch
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
from vision.datasets.coco_dataset import CustomCOCO
from torch.utils.data import DataLoader
import os
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_custom_ssd_lite, create_custom_ssd_lite_predictor
from vision.utils.info_loss import *

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset_type", default="COCO", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


# def group_annotation_by_class(dataset):
#     true_case_stat = {}
#     all_gt_boxes = {}
#     all_difficult_cases = {}
#     for i in range(len(dataset)):
#         image_id, annotation = dataset.get_annotation(i)
#         gt_boxes, classes, is_difficult = annotation
#         gt_boxes = torch.from_numpy(gt_boxes)
#         for i, difficult in enumerate(is_difficult):
#             class_index = int(classes[i])
#             gt_box = gt_boxes[i]
#             if not difficult:
#                 true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1
#
#             if class_index not in all_gt_boxes:
#                 all_gt_boxes[class_index] = {}
#             if image_id not in all_gt_boxes[class_index]:
#                 all_gt_boxes[class_index][image_id] = []
#             all_gt_boxes[class_index][image_id].append(gt_box)
#             if class_index not in all_difficult_cases:
#                 all_difficult_cases[class_index]={}
#             if image_id not in all_difficult_cases[class_index]:
#                 all_difficult_cases[class_index][image_id] = []
#             all_difficult_cases[class_index][image_id].append(difficult)
#
#     for class_index in all_gt_boxes:
#         for image_id in all_gt_boxes[class_index]:
#             all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
#     for class_index in all_difficult_cases:
#         for image_id in all_difficult_cases[class_index]:
#             all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
#     return true_case_stat, all_gt_boxes, all_difficult_cases
#
#
# def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
#                                         prediction_file, iou_threshold, use_2007_metric):
#     with open(prediction_file) as f:
#         image_ids = []
#         boxes = []
#         scores = []
#         for line in f:
#             t = line.rstrip().split(" ")
#             image_ids.append(t[0])
#             scores.append(float(t[1]))
#             box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
#             box -= 1.0  # convert to python format where indexes start from 0
#             boxes.append(box)
#         scores = np.array(scores)
#         sorted_indexes = np.argsort(-scores)
#         boxes = [boxes[i] for i in sorted_indexes]
#         image_ids = [image_ids[i] for i in sorted_indexes]
#         true_positive = np.zeros(len(image_ids))
#         false_positive = np.zeros(len(image_ids))
#         matched = set()
#         for i, image_id in enumerate(image_ids):
#             box = boxes[i]
#             if image_id not in gt_boxes:
#                 false_positive[i] = 1
#                 continue
#
#             gt_box = gt_boxes[image_id]
#             ious = box_utils.iou_of(box, gt_box)
#             max_iou = torch.max(ious).item()
#             max_arg = torch.argmax(ious).item()
#             if max_iou > iou_threshold:
#                 if difficult_cases[image_id][max_arg] == 0:
#                     if (image_id, max_arg) not in matched:
#                         true_positive[i] = 1
#                         matched.add((image_id, max_arg))
#                     else:
#                         false_positive[i] = 1
#             else:
#                 false_positive[i] = 1
#
#     true_positive = true_positive.cumsum()
#     false_positive = false_positive.cumsum()
#     precision = true_positive / (true_positive + false_positive)
#     recall = true_positive / num_true_cases
#     if use_2007_metric:
#         return measurements.compute_voc2007_average_precision(precision, recall)
#     else:
#         return measurements.compute_average_precision(precision, recall)

def test(dataset, predictor):
    # TODO : want to make {key:class -> val: [mean precision per each class, mean IOU per each class]}
    result_table = {}
    true_positive = np.zeros(len(dataset))
    false_positive = np.zeros(len(dataset))
    for i in range(len(dataset)):
        img, gt_boxes, gt_labels = dataset[i]   # img = (B, 3, 300, 300). At test, B=1
        boxes, labels, probs = predictor.predict(img)

        print('Predicted ::', boxes.shape, labels.shape, probs.shape) # (M,4), (M,1), (M)
        print('GT ::', gt_boxes.shape, gt_labels.shape)      # (N, 4) , (N, 1)
        # N, M 은 가변적임.
        # (1) TODO : ground truth에 해당하는 class만 predicted labels에서 그 인덱스를 추출.
        # (2) TODO : 추출한 인덱스를 이용해 box coordinate로 GT box와 IOU 계산 :: box_utils에 있는 함수 이용 또는 숙제로 구현한 함수 이용
        # (3) TODO : IOU가 최대가 되도록 하는 인덱스만 골라냄.
        # (4) TODO : result_table 의 key:label에 최종 인덱스 결과의 박스 좌표, prob값을 넣어줌.

        # (1)
        for i, gt_label in enumerate(gt_labels):
            pred_idx = []
            for j, label in enumerate(labels):
                if label[0] == gt_label[0]:
                    pred_idx.append(j)
            pred_box_ = boxes[pred_idx]

            # (2) & (3)
            # TODO: predicted label에 GT label이 없는 경우도 처리해야 함.

            ious = box_utils.iou_of(pred_box_, gt_boxes[i])
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > args.iou_threshold:
                true_positive[i] = 1
                prob_ = probs[max_arg]
            else:
                false_positive[i] = 1

            # (4)

    return result_table


def test_report(result_table):
    mAP = 0
    mIOU = 0

    # TODO : by using result_table, caculate mAP and mIOU.
    # reference : vision/utils/measurement.py

    return mAP, mIOU


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == 'COCO':
        # TODO : COCO dataset -> Done
        dataset = CustomCOCO(args.dataset,
                             mode=1)
    else:
        raise NotImplementedError('Other Datasets are not supported')

    # true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
    elif args.net == 'custom':
        net = create_custom_ssd_lite(len(class_names), is_test=True)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')

    if args.net == 'mb2-ssd-lite' or args.net == "mb3-large-ssd-lite" or args.net == "mb3-small-ssd-lite":
        predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    elif args.net == 'custom':
        # TODO -> Done
        predictor = create_custom_ssd_lite_predictor(net, nms_method=args.nms_method, device=DEVICE)
    else:
        logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    result = test(dataset, predictor)
