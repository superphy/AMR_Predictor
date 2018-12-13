#Script to create figures for various types of results

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

#Function to create an MIC frequency bar graph
def make_grdi_MIC_bar_graph():
    drugs = ['AMC', 'AMP', 'AZM', 'CHL', 'CIP', 'CRO', 'FOX', 'SXT', 'TET', 'TIO', 'GEN', 'NAL']
    for drug in drugs:
        data_frame = joblib.load("../data/dataframes/grdi_" + drug + "_df_mic.pkl")
        avg_report = joblib.load("../data/avg_reports/grdi_" + drug + "_df_reports.pkl")
        recall = avg_report["Recall"]

        #plot the results of the DataFrame
        sns.set(style="whitegrid")
        ax = sns.barplot(x="MIC(mg/L)", y="No. Genomes", data=data_frame)
        ax.set_title(drug+' GRDI MIC Frequencies')

        #for loop to layer values on top of the bars in the graph
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, '{:1.2f}'.format(recall[i]), ha="center")

        #save and export the figure
        ax.figure.savefig("../Figures/grdi_mic_figures_"+drug+".png")
        plt.clf() #clear plot for the next figure


def make_public_MIC_bar_graph():
    drugs = ['AMC', 'AMP', 'AZM', 'CHL', 'CIP', 'CRO', 'FIS', 'FOX', 'SXT', 'TET', 'TIO', 'GEN', 'NAL']
    for drug in drugs:
        data_frame = joblib.load("../data/dataframes/public_" + drug + "_df_mic.pkl")
        avg_report = joblib.load("../data/avg_reports/public_" + drug + "_df_reports.pkl")
        recall = avg_report["Recall"]

        #plot the results of the DataFrame
        sns.set(style="whitegrid")
        ax = sns.barplot(x="MIC(mg/L)", y="No. Genomes", data=data_frame)
        ax.set_title(drug+' Public MIC Frequencies')

        #for loop to layer values on top of the bars in the graph
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, '{:1.2f}'.format(recall[i]), ha="center")

        #save and export the figure
        ax.figure.savefig("../Figures/public_mic_figures_"+drug+".png")
        plt.clf() #clear plot for the next figure


#!/usr/bin/env python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
import os, sys

if __name__ == "__main__":
    path = sys.argv[1]
    num_samples = sum([len(files) for r, d, files in os.walk(path)])

    master_df = pd.DataFrame(data = np.zeros((num_samples, 7),dtype = 'object'),index =[i for i in range(num_samples)], columns = ['acc','model','feats','train','test','attribute', '1Dacc'])
    title_string = ''
    for root, dirs, files in os.walk(path):
        for i, file in enumerate(files):
            file = file.split('.')[0]
            attribute, num_feats, train, test = file.split('_')
            num_feats = int(num_feats[:-5])
            model = train[:3]
            train = train[12:]
            test = test[8:]
            #print(attribute, num_feats, model, train, test)
            acc_data = pd.read_pickle(path+file+'.pkl')
            acc = 0
            total = np.sum(acc_data.values[:,3])
            for row in acc_data.values:
                acc += row[1]*row[3]/total
            for j, stat in enumerate([acc,model,num_feats,train,test,attribute]):
                master_df.values[i,j] = stat
                master_df.values[i,7] = acc_data.values[0,5]
            if(i==0):
                title_string = (("{} predictor trained on {}, tested on {}".format(attribute, train, test)))

    master_df['feats'] = pd.to_numeric(master_df['feats'])
    master_df['acc'] = pd.to_numeric(master_df['acc'])
    print(master_df)
    print(master_df.dtypes)
    base = sns.relplot(x="feats", y="acc", hue="model", kind="line", data=master_df)\
    plt.rcParams["axes.titlesize"] = 8
    plt.title(title_string + ' (Direct)')
    plt.ylim(0,1)
    plt.savefig('figures/0D_'+(title_string.replace(" ",""))+'.png')

    window = sns.relplot(x="feats", y="1Dacc", hue="model", kind="line", data=master_df)
    plt.rcParams["axes.titlesize"] = 8
    plt.title(title_string + ' (1-Dilution)')
    plt.ylim(0,1)
    plt.savefig('figures/1D_'+(title_string.replace(" ",""))+'.png')
