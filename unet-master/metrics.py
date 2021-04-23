import keras.backend as K
import numpy as np
#
# # 自定义评价函数1
def iou_keras(y_true, y_pred):
    """
    Return the Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the IoU for the given label
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(y_pred, label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou_keras(y_true, y_pred):
    """
    Return the mean Intersection over Union (IoU).
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the mean IoU
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())

    mean_iou = K.variable(0)

    thre_list = list(np.arange(0.0000001, 0.99, 0.05))

    for thre in thre_list:
        y_pred_temp = K.cast(y_pred >= thre, K.floatx())
        y_pred_temp = K.cast(K.equal(y_pred_temp, label), K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred_temp)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred_temp) - intersection
        iou = K.switch(K.equal(union, 0), 1.0, intersection / union)
        mean_iou = mean_iou + iou

    return mean_iou / len(thre_list)


# 第二种
# def iou(y_true, y_pred):
#     """
#     Return the Intersection over Union (IoU).
#     Args:
#         y_true: the expected y values as a one-hot
#         y_pred: the predicted y values as a one-hot or softmax output
#     Returns:
#         the IoU for the given label
#     """
#     label = 1
#     # extract the label values using the argmax operator then
#     # calculate equality of the predictions and truths to the label
#     y_true = K.cast(K.equal(y_true, label), K.floatx())
#     y_pred = K.cast(K.equal(y_pred, label), K.floatx())
#     # calculate the |intersection| (AND) of the labels
#     intersection = K.sum(y_true * y_pred)
#     # calculate the |union| (OR) of the labels
#     union = K.sum(y_true) + K.sum(y_pred) - intersection
#     # avoid divide by zero - if the union is zero, return 1
#     # otherwise, return the intersection over union
#     return K.switch(K.equal(union, 0), 1.0, intersection / union)