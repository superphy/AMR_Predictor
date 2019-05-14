"""
This serves to average the results of the independent
hyperas tests, for internal use by hyperas.smk
"""

import numpy as np
import pandas as pd
import os, sys

if __name__ =="__main__":
    feats = sys.argv[1]
    drug  = sys.argv[2]
    dataset = sys.argv[3]

    if(dataset != 'public'):
        raise Exception("hyp_average.py is not yet setup for non public data")

    path=''
    if(dataset=='grdi'):
        path = 'grdi_'
    elif(dataset=='kh'):
        path = 'kh_'

    OBN_accs = []
    OBO_accs = []

    index = []
    final = []

    # everything is saved in data/{path}{drug}/hyperas/
    for i in range(1,6):
        split_df = pd.read_pickle("data/"+path+drug+"/hyperas/"+str(feats)+"feats_"+str(i)+".pkl")

        # initialize new dataframe values
        if i==1:
            final = np.zeros((split_df.shape),dtype=float)
            index = split_df.index

        num_samples = np.sum(split_df.values[:,3])

        # find direct accuracy
        running_sum = 0
        for row in split_df.values:
            running_sum+=(row[1]*row[3]/num_samples)

        # append splits 1-D and direct accuracies to be saved
        OBN_accs.append(running_sum)
        OBO_accs.append(split_df.values[0,4])

        # add values to new dataframe
        for i, row in enumerate(split_df.values):
            for j, cell in enumerate(row):
                final[i,j] += cell

    # average out the cells
    for i, row in enumerate(split_df.values):
        for j, cell in enumerate(row):
            if j == 3:
                continue
            final[i,j] /= 5


    final_df = pd.DataFrame(data = final, index = index, columns = ['Precision','Recall', 'F-Score','Supports', '1D Acc'])

    final_df.to_pickle("results/public1_"+drug+"/"+drug+"_"+feats+"feats_ANNtrainedOnpublic_testedOnaCrossValidation.pkl")

    if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/split_accuracies"):
        os.mkdir(os.path.abspath(os.path.curdir)+"/data/split_accuracies")
    # saving the accuracies for each split
    np.save('data/split_accuracies/'+drug+'_'+str(feats)+'feats_ANNtrainedOnpublic_testedOnaCrossValidation.npy' ,np.vstack((OBN_accs,OBO_accs)))
