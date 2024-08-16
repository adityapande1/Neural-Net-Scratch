import numpy as np


# precision implementation
def calculate_precision(y_true, y_scores, threshold=0.5):
    y_pred = []
    for sample in y_scores:
        y_pred.append([1 if i >= threshold else 0 for i in sample])
    y_pred = np.array(y_pred)
    pr = []
    for y, y_hat in zip(y_true, y_pred):
        true_set = set(np.where(y == 1)[0] + 1)
        pred_set = set(np.where(y_hat == 1)[0] + 1)
        intersection = set.intersection(true_set, pred_set)
        num_labels_pred_set = len(pred_set)
        if num_labels_pred_set <= 0 and len(true_set) <= 0:
            pr.append(1)
        elif num_labels_pred_set <= 0 and len(true_set) > 0:
            pr.append(0)
        else:
            pr.append(len(intersection) / num_labels_pred_set)
    return np.mean(pr)


# recall implementation
def calculate_recall(y_true, y_scores, threshold=0.5):
    y_pred = []
    for sample in y_scores:
        y_pred.append([1 if i >= threshold else 0 for i in sample])
    y_pred = np.array(y_pred)
    rc = []
    for y, y_hat in zip(y_true, y_pred):
        true_set = set(np.where(y == 1)[0] + 1)
        pred_set = set(np.where(y_hat == 1)[0] + 1)
        intersection = set.intersection(true_set, pred_set)
        num_labels_true_set = len(true_set)
        if num_labels_true_set <= 0 and len(pred_set) <= 0:
            rc.append(1)
        elif num_labels_true_set <= 0 and len(pred_set) > 0:
            rc.append(0)
        else:
            rc.append(len(intersection) / num_labels_true_set)
    return np.mean(rc)
