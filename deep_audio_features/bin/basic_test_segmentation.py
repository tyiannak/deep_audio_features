import argparse
from torch.utils.data import DataLoader
import sys
import os
import pickle
import deep_audio_features.bin.basic_test as daf_test
import deep_audio_features.bin.config
import numpy as np
import csv
import itertools
import sklearn.metrics as metrics
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.models.cnn import load_cnn


GT_RESOLUTION = 0.1


def upsample_sequence(seq, in_segment_duration, out_segment_duration):
    times = int(in_segment_duration / out_segment_duration)
    out = list(itertools.chain.from_iterable(itertools.repeat(x, times) for x in seq))
    return out


def segments_to_labels(start_times, end_times, gt_labels, window):
    """
    This function converts segment endpoints and respective segment
    gt_labels to fix-sized class gt_labels.
    ARGUMENTS:
     - start_times:  segment start points (in seconds)
     - end_times:    segment endpoints (in seconds)
     - gt_labels:       segment gt_labels
     - window:      fix-sized window (in seconds)
    RETURNS:
     - flags:    np array of class indices
     - gt_class_names:    list of classnames (strings)
    """
    flags = []
    gt_class_names = list(set(gt_labels))
    # TODO test for multiclass
    if len(gt_class_names)==1:
        gt_class_names.append("non" + gt_class_names[0])
    index = window / 2.0
    found = False
    while index < end_times[-1]:
        for i in range(len(start_times)):
            if start_times[i] < index <= end_times[i]:
                found = True
                break
        if found:
            flags.append(gt_class_names.index(gt_labels[i]))
        else:
            flags.append(len(gt_class_names)-1)
        found = False
        index += window
    print(flags), gt_class_names
    return np.array(flags), gt_class_names


def read_segmentation_gt(gt_file):
    """
    This function reads a segmentation ground truth file,
    following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>
    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a np array of segments' start positions
     - seg_end:       a np array of segments' ending positions
     - seg_label:     a list of respective class gt_labels (strings)
    """
    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter='\t')
        start_times = []
        end_times = []
        gt_labels = []
        for row in reader:
            if len(row) == 3:
                start_times.append(float(row[0]))
                end_times.append(float(row[1]))
                gt_labels.append((row[2]))
    return np.array(start_times), np.array(end_times), gt_labels


def load_ground_truth_segments(gt_file, mt_step):
    seg_start, seg_end, seg_labels = read_segmentation_gt(gt_file)
    gt_labels, gt_class_names = segments_to_labels(seg_start, seg_end,
                                                   seg_labels, mt_step)
    labels_temp = []
    for index, label in enumerate(gt_labels):
        # "align" gt_labels with GT
        if gt_class_names[gt_labels[index]] in gt_class_names:
            labels_temp.append(gt_class_names.index(gt_class_names[
                                                     gt_labels[index]]))
        else:
            labels_temp.append(-1)
    gt_labels = np.array(labels_temp)

    return gt_labels, gt_class_names


if __name__ == '__main__':

    # Read arguments -- model
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True,
                        type=str, help='Model')

    parser.add_argument('-i', '--input', required=True,
                        type=str, help='Input file for testing')

    parser.add_argument('-g', '--groundtruth', required=True,
                        type=str, help='ground truth file for testing')

    parser.add_argument('-s', '--segmentation', required=False,
                        action='store_true',
                        help='Return segment predictions')

    args = parser.parse_args()

    # Get arguments
    model_name = args.model
    ifile = args.input
    seg = args.segmentation

    # Restore model
    with open(model_name, "rb") as input_file:
        model_params = pickle.load(input_file)
    if "classes_mapping" in model_params:
        task = "classification"
        model, hop_length, window_length = load_cnn(model_name)
        class_names_model = model.classes_mapping

    # segment-level predictions using the model:
    labels, p = daf_test.test_model(modelpath=model_name, ifile=ifile,
                                    layers_dropped=0, test_segmentation=seg)

    # load the ground truth file: 
    gt_labels, gt_class_names = load_ground_truth_segments(args.groundtruth, 
                                                           GT_RESOLUTION)
    seg_size = ((model_params["spec_size"])[1] - 1) * model_params["window_length"]

    labels2 = upsample_sequence(labels, seg_size, GT_RESOLUTION)

    # cut last segments:
    min_len = min(len(labels2), len(gt_labels))
    labels2 = labels2[:min_len]
    gt_labels = gt_labels[:min_len]

    #for i in range(len(gt_labels)):
    #    print(gt_class_names[gt_labels[i]], class_names_model[d2[i]])
    # convert label ids to labels (both for results and ground truth):
    results_gt = [gt_class_names[gt_labels[i]] for i in range(len(gt_labels))]
    results = [class_names_model[labels2[i]] for i in range(len(labels2))]
    overall_accuracy = metrics.accuracy_score(results_gt, results)
    print(f'Accuracy: {overall_accuracy:.2f}')
    print(metrics.recall_score(results_gt, results, average=None))
    print(metrics.precision_score(results_gt, results, average=None))

    time = np.arange(0, len(results) * GT_RESOLUTION, GT_RESOLUTION)

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(go.Scatter(x=time, y=results),  row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=results_gt), row=2, col=1)
    fig.update_layout(height=600, title_text="Side By Side Subplots")
    fig.show()