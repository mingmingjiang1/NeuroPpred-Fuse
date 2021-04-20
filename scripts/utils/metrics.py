import numpy as np
import math



    
    
    
def calculate_confusion_matrix(y_true, y_pred):#�������������ļ���,y_true����ʵ��Ԥ���ǩ��y_pred��Ԥ��ı�ǩ��shape����1d��
    corrects = 0
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(y_true)):
        confusion_matrix[y_pred[i]][y_true[i]] += 1

        if y_true[i] == y_pred[i]:
            corrects = corrects + 1

    acc = corrects * 1.0 / len(y_true)
    specificity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    sensitivity = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1])
    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    if math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) == 0: mcc = 0
    else: mcc = (tp * tn - fp * fn ) / math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    return acc, confusion_matrix, sensitivity, specificity, mcc
