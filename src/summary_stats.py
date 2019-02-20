#!/usr/bin/env python

'''
This script is intended to summarize the results of run_tests.py/run_tests_slurm.py
'''

import os, sys
import pandas as pd
import numpy as np
import glob

path = "results/"
num_samples = sum([len(files) for r, d, files in os.walk(path)])

print("found {} samples in {}".format(num_samples, path))

master_df = pd.DataFrame(data = np.zeros((num_samples, 7),dtype = 'object'),index =[i for i in range(num_samples)], columns = ['acc','model','feats','train','test','attribute', '1Dacc'])
title_string = ''
counter = 0
directs = [[root,files] for root, dirs, files in os.walk(path) if not dirs]
for i in directs:
    direc = i[0]
    for k, file in enumerate(i[1]):
        filepath = direc+'/'+file

        file = file.split('.')[0]
        attribute, num_feats, train, test = file.split('_')
        num_feats = int(num_feats[:-5])
        model = train[:3]
        train = train[12:]
        test = test[8:]
        #print(attribute, num_feats, model, train, test)
        acc_data = pd.read_pickle(filepath)
        acc = 0
        total = np.sum(acc_data.values[:,3])
        for row in acc_data.values:
            acc += row[1]*row[3]/total
        for j, stat in enumerate([acc,model,num_feats,train,test,attribute]):
            #print(j,stat)
            #print(master_df.values.shape)
            master_df.values[counter,j] = stat
            master_df.values[counter,6] = acc_data.values[0,4]
        if(i==0):
            title_string = (("{} predictor trained on {}, tested on {}".format(attribute, train, test)))
        counter+=1
print(counter)
master_df['feats'] = pd.to_numeric(master_df['feats'])
master_df['acc'] = pd.to_numeric(master_df['acc'])
master_df['1Dacc'] = pd.to_numeric(master_df['1Dacc'])

#pd.set_option('display.max_rows', 500)
print(master_df)

pub_count = 0
pub_acc = 0
grdi_count = 0
grdi_acc = 0

for row in master_df.values:
    if(row[1]== 'XGB' and row[2]==3000 and row[3]=='public' and row[4]=='aCrossValidation'):
        pub_acc+=row[0]
        pub_count+=1
    if(row[1]== 'XGB' and row[2]==3000 and row[3]=='grdi' and row[4]=='aCrossValidation'):
        grdi_acc+=row[0]
        grdi_count+=1
print(pub_acc/pub_count)
print(grdi_acc/grdi_count)


"""
filled = 0
for i in master_df['acc']:
    if i > 0.001:
        filled+=1
print(filled)

drugs   = ["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
models  = ["SVM", "ANN", "XGB"]
dataset = ["public1","grdi2","kh3","public4_grdi","grdi5_public","grdi6_kh","kh7_grdi"]


"""
