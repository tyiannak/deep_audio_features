import argparse
import torch
from torch.utils.data import DataLoader
import sys, os
import pickle
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../"))
from deep_audio_features.models.cnn import load_cnn
import deep_audio_features.bin.basic_test
import deep_audio_features.bin.config
import numpy as np
import csv

def segments_to_labels(start_times, end_times, labels, window):
    """
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - start_times:  segment start points (in seconds)
     - end_times:    segment endpoints (in seconds)
     - labels:       segment labels
     - window:      fix-sized window (in seconds)
    RETURNS:
     - flags:    np array of class indices
     - class_names:    list of classnames (strings)
    """
    flags = []
    class_names = list(set(labels))
    # TODO test for multiclass
    if len(class_names)==1:
        class_names.append("non" + class_names[0])
    index = window / 2.0
    found = False
    while index < end_times[-1]:
        for i in range(len(start_times)):
            if start_times[i] < index <= end_times[i]:
                found = True
                break
        if found:
            flags.append(class_names.index(labels[i]))
        else:
            flags.append(len(class_names)-1)
        found = False
        index += window
    print(flags), class_names
    return np.array(flags), class_names

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
     - seg_label:     a list of respective class labels (strings)
    """
    with open(gt_file, 'rt') as f_handle:
        reader = csv.reader(f_handle, delimiter='\t')
        start_times = []
        end_times = []
        labels = []
        for row in reader:
            if len(row) == 3:
                start_times.append(float(row[0]))
                end_times.append(float(row[1]))
                labels.append((row[2]))
    return np.array(start_times), np.array(end_times), labels

def load_ground_truth_segments(gt_file, mt_step):
    seg_start, seg_end, seg_labels = read_segmentation_gt(gt_file)
    #print(unique(seg_labels))
    labels, class_names = segments_to_labels(seg_start, seg_end, seg_labels,
                                             mt_step)
    labels_temp = []
    for index, label in enumerate(labels):
        # "align" labels with GT
        if class_names[labels[index]] in class_names:
            labels_temp.append(class_names.index(class_names[
                                                     labels[index]]))
        else:
            labels_temp.append(-1)
    labels = np.array(labels_temp)

    return labels, class_names

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

    parser.add_argument('-L', '--layers', required=False, default=0,
                        help='Number of final layers to cut. Default is 0.')
    args = parser.parse_args()

    # Get arguments
    model_name = args.model
    ifile = args.input
    layers_dropped = int(args.layers)
    segmentation = args.segmentation

    # Restore model
    with open(model_name, "rb") as input_file:
        model_params = pickle.load(input_file)
    if "classes_mapping" in model_params:
        task = "classification"
        model, hop_length, window_length = load_cnn(model_name)
        class_names_model = model.classes_mapping

    # Test the model
    d, p = deep_audio_features.bin.basic_test.test_model(modelpath=model_name, 
                      ifile=ifile,
                      layers_dropped=layers_dropped,
                      test_segmentation=segmentation)


    labels, class_names = load_ground_truth_segments(args.groundtruth, 0.1)
    for i in range(len(labels)):
        print(i, class_names[labels[i]])

    for i in range(len(d)):
        print(class_names_model[d[i]])

    seg_size = ((model_params["spec_size"])[1] - 1) * 0.05

    import itertools
    times = int(seg_size / 0.1)
    d2 = list(itertools.chain.from_iterable(itertools.repeat(x, times) for x in d))

    min_len = min(len(d2), len(labels))
    d2 = d2[:min_len]
    for i in range(len(d2)):
        if d2[i] == 3:
            d2[i] = 0
            print("AAAA")
        else:
            d2[i] = 1
    labels = labels[:min_len]

    print(class_names_model)
    print(class_names)
    print(labels)
    print(d2)
    import sklearn.metrics as metrics
    print(metrics.accuracy_score(labels, d2))
    print(metrics.recall_score(labels, d2, average=None))
    print(metrics.precision_score(labels, d2, average=None))
