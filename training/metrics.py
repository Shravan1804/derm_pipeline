#!/usr/bin/env python

"""metrics.py: Training metrics"""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))

def get_cls_TP_TN_FP_FN(cls_truth, cls_preds):
    """Computes true positives, true negatives, false positives, false negatives
    :param cls_truth: tensor of bool, targets matching category
    :param cls_preds: tensor of bool, decoded predictions matching category
    :return: tuple TP, TN, FP, FN
    """
    TP = (cls_preds & cls_truth).sum().item()
    TN = (~cls_preds & ~cls_truth).sum().item()
    FP = (cls_preds & ~cls_truth).sum().item()
    FN = (~cls_preds & cls_truth).sum().item()
    return TP, TN, FP, FN


def accuracy(TP, TN, FP, FN, epsilon=1e-8):
    """Accuracy metric
    # Make sure classes are balanced
    # Proportion of both Positive and Negative that were correctly classified
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, accuracy result
    """
    return (TP + TN) / (TP + TN + FP + FN + epsilon)


def precision(TP, TN, FP, FN, epsilon=1e-8):
    """Precision metric
    # Proportion of predicted Positives that are truly Positive
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, precision result
    """
    return TP / (TP + FP + epsilon)


def recall(TP, TN, FP, FN, epsilon=1e-8):
    """Recall metric
    # Proportion of actual Positives (in ground truth) that are correctly classified
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, recall result
    """
    return TP / (TP + FN + epsilon)


def F1(TP, TN, FP, FN, epsilon=1e-8):
    """Harmonic mean of precision and recall
    # Can be used to compare two classifiers BUT
    # F1-score gives a larger weight to lower numbers e.g. 100% pre and 0% rec => 0% F1
    # F1-score gives equal weight to pre/rec which may not what we seek depending on the problem
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, F1 result
    """
    pre, rec = precision(TP, TN, FP, FN, epsilon), recall(TP, TN, FP, FN, epsilon)
    return 2 * pre * rec / (pre + rec)


def specificity(TP, TN, FP, FN, epsilon=1e-8):
    """Specificity metric
    # Proportion of true negative given gt is false
    # Proba negative test given patient is not sick
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, specificity result
    """
    return TN / (TN + FP + epsilon)


def sensitivity(TP, TN, FP, FN, epsilon=1e-8):
    """Sensitivity metric
    # Proportion true positive given gt is true (i.e. RECALL)
    # Proba positive test given patient is sick
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, sensitivity result
    """
    return TP / (TP + FN + epsilon)


def ppv(TP, TN, FP, FN, epsilon=1e-8):
    """Positive predictive value metric
    # Proportion of positive predictions that are true positives (i.e. PRECISION)
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, ppv result
    """
    return TP / (TP + FP + epsilon)


def npv(TP, TN, FP, FN, epsilon=1e-8):
    """Negative predictive value metric
    # Proportion of negative predictions that are true negatives
    :param TP: int, number of true positives
    :param TN: int, number of true negatives
    :param FP: int, number of false positives
    :param FN: int, number of false negatives
    :param epsilon: float, prevents division by zero
    :return: float, npv result
    """
    return TN / (TN + FN + epsilon)