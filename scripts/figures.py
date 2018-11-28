#Script to create figures for various types of results

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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


def make_public_MIC_bar_graph(data_frame, drug, avg_reports):
    #plot the results of the DataFrame
    sns.set(style="whitegrid")
    ax = sns.barplot(x="MIC", y="Count", data=data_frame)
    ax.set_title(drug+' MIC Frequencies')


    #for loop to layer values on top of the bars in the graph
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + 3, '{:1.2f}'.format(avg_reports[i][1]), ha="center")

    #save and export the figure
    ax.figure.savefig("Figures/mic_figures_"+drug+".png")
    plt.clf() #clear plot for the next figure

