import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os, sys

PATRIC_res = pd.read_excel("data/PATRIC_pred_act.xlsx", skiprows=1)

our_mic = ['AMP','AMC','CRO','AZM','CHL','CIP','SXT','FIS','FOX','GEN','NAL','TET','TIO']
pat_mic = ['AMP','AUG','AXO','AZI','CHL','CIP','COT','FIS','FOX','GEN','NAL','TET','TIO']

def standard_mic(mic):
    """
    takes in an MIC from the patric supp tables and converts it to a standard
    format
    """
    val = str(mic).lstrip('<>=').rstrip('.0')
    if isinstance(mic, float):
        return val
    if ('>' in mic or '<' in mic) and '=' not in mic:
        return standard_mic(2*float(val))
    else:
        return val


for drug_pos, drug in enumerate(pat_mic):
    drug_df = PATRIC_res[PATRIC_res['Antibiotic'] == drug]

    actual = [standard_mic(i) for i in drug_df["Laboratory-derived MIC"]]
    predicted = [standard_mic(i) for i in drug_df["Predicted MIC"]]

    labels = list(set(actual).union(set(predicted)))

    labels = sorted(labels, key=float)

    data = np.transpose(precision_recall_fscore_support(actual, predicted, average=None, labels = labels))

    result_df = pd.DataFrame(data = data, index = labels, columns=['Precision','Recall', 'F-Score','Supports'])
    print(result_df)

    errors_list = []

    total = len(actual)
    cor = 0
    obo = 0
    for act, pred in zip(actual, predicted):
        act = float(act)
        pred = float(pred)
        if(act in [8,16]):
            print(act,pred)
        if act == pred:
            cor +=1
        elif(pred*2 == act or pred/2 == act):
            obo +=1
        else:
            errors_list.append((act,pred))

    print("Direct: ", cor/total)
    print("1D-Acc: ", (cor+obo)/total)

    # print this to check to make sure no correct are being missed
    #print(set(errors_list))


    sys.exit()
