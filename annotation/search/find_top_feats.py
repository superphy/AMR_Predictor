def find_top_feats(imp_arr,sav_path,top_x):
    """
    Takes in a 2-d importance array from model.py and returns the top X features
    """
    top_feats = []
    while(top_x>0):
        # find the highest scoring value
        m = max(imp_arr[1])

        # pull the index of all kmers tieing the top score
        top_indeces = [i for i, j in enumerate(imp_arr[1]) if j == m]

        # make sure we dont take more than top_x total, this can be removed
        # to keep all tying features in the pipeline but beware, if all are zero it will search through a thousand kmers
        if(len(top_indeces) > top_x):
            top_indeces = top_indeces[:top_x]

        top_x -= len(top_indeces)
        for i in top_indeces:
            top_feats.append(imp_arr[0][i])
            imp_arr[1][i] = 0
    return top_feats

if __name__ =="__main__":
    import os, sys
    import numpy as np
    import math

    # where we load the features from
    input_arr = sys.argv[1]

    # where we intend on saving the top 5
    output = sys.argv[2]
    filename = output.split("/")[-1]
    save_loc = output[:-len(filename)]

    # how many features to return
    top_x = sys.argv[3]

    if not os.path.exists(os.path.abspath(os.path.curdir)+"/"+save_loc):
        os.mkdir(os.path.abspath(os.path.curdir)+"/"+save_loc)


    if('grdi' in input_arr):#and 'AZM' not in input_arr):
        print("Merging 5 fold cross validation")
        input_arr = input_arr[:-14] + 'cv_fold1.npy'
        feat_imps = np.load(input_arr, allow_pickle=True)
        for i in range(2,6):
            for j in feat_imps[1]:
                try:
                    assert(not math.isnan(float(j.decode('utf-8'))))
                except:
                    print("fold {} has nan values")
                    break
            fold_arr = input_arr[:-5] + str(i) + '.npy'
            feat_imps = np.hstack((feat_imps,np.load(fold_arr, allow_pickle=True)))
    else:
        print("Using features trained from whole dataset")
        feat_imps = np.load(input_arr, allow_pickle=True)


    if(isinstance(feat_imps[0,0],bytes)):
        feat_imps = [[i.decode('utf-8') for i in feat_imps[0]],[float(i.decode('utf-8')) for i in feat_imps[1]]]

    for i in feat_imps[1]:
        assert(not math.isnan(i))

    top_feats = find_top_feats(feat_imps,output,int(top_x))

    np.save(output, top_feats)
