#Script to create figures for various types of results

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

#Function to create an MIC frequency bar graph
def make_MIC_bar_graph(data_frame, drug, avg_reports):
    #plot the results of the DataFrame
    sns.set(style="whitegrid")
    ax = sns.barplot(x="MIC", y="Count", data=data_frame)
    ax.set_title(drug+' MIC Frequencies')


    #for loop to layer values on top of the bars in the graph
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, '{:1.2f}'.format(avg_reports[i][1]), ha="center")

    #save and export the figure
    ax.figure.savefig("Figures/grdi_mic_figures_"+drug+".png")
    plt.clf() #clear plot for the next figure


def make_public_MIC_bar_graph():
    drugs = ['AMC', 'AMP', 'AZM', 'CHL', 'CIP', 'CRO', 'FIS', 'FOX', 'SXT', 'TET', 'TIO', 'GEN', 'NAL']
    for drug in drugs:
        data_frame = joblib.load("../data/dataframes/" + drug + "_df_mic.pkl")
        avg_report = joblib.load("../data/avg_reports/" + drug + "_df_reports.pkl")
        recall = avg_report["Recall"]
        #plot the results of the DataFrame
        sns.set(style="whitegrid")
        ax = sns.barplot(x="MIC", y="Count", data=data_frame)
        ax.set_title(drug+' MIC Frequencies')


        #for loop to layer values on top of the bars in the graph
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, '{:1.2f}'.format(recall[i]), ha="center")

        #save and export the figure
        ax.figure.savefig("../Figures/public_mic_figures_"+drug+".png")
        plt.clf() #clear plot for the next figure

