
#!/usr/bin/env python

"""bin_mics.py
Convert MIC values into distinct classes. Save processed
classes as pandas dataframes.
"""

import os
import logging
import numpy as np
import pandas as pd
import sys

from dotenv import find_dotenv, load_dotenv
from sklearn.externals import joblib

from mic import MICPanel, MGPL

__author__ = "Matthew Whiteside"
__copyright__ = "Copyright 2018, Public Health Agency of Canada"
__license__ = "APL"
__version__ = "2.0"
__maintainer__ = "Matthew Whiteside"
__email__ = "matthew.whiteside@phac-aspc.gc.ca"

# Manually define MIC ranges due to mixing of different systems
mic_ranges = {
    'MIC_AMP': {
        'top': '>32.0000',
        'bottom': '<=1.0000',
    },
    'MIC_AMC': {
        'top': '>32.0000',
        'bottom': '<=1.0000',
    },
    'MIC_FOX': {
        'top': '>32.0000',
        'bottom': '1.0000',
    },
    'MIC_CRO': {
        'top': '64.0000',
        'bottom': '<=0.2500',
    },
    'MIC_TIO': {
        'top': '>8.0000',
        'bottom': '0.2500',
    },
    'MIC_GEN': {
        'top': '>16.0000',
        'bottom': '<=0.2500',
    },
    'MIC_FIS': {
        'top': '>256.0000',
        'bottom': '<=16.0000',
    },
    'MIC_SXT': {
        'top': '>64.0000',
        'bottom': '<=0.1250',
    },
    'MIC_AZM': {
        'top': '>16.0000',
        'bottom': '<=1.0000',
    },
    'MIC_CHL': {
        'top': '>32.0000',
        'bottom': '<=2.0000',
    },
    'MIC_CIP': {
        'top': '>4.0000',
        'bottom': '<=0.0150',
    },
    'MIC_NAL': {
        'top': '>32.0000',
        'bottom': '1.0000',
    },
    'MIC_TET': {
        'top': '>32.0000',
        'bottom': '<=4.0000',
    },

}
'''
mic_ranges = {
    'MIC_AMP': {
        'top': '>=32.0000',
        'bottom': '<=1.0000',
    },
    'MIC_AMC': {
        'top': '>=32.0000',
        'bottom': '<=1.0000',
    },
    'MIC_FOX': {
        'top': '>=32.0000',
        'bottom': '<=1.0000',
    },
    'MIC_CRO': {
        'top': '>=64.0000',
        'bottom': '<=0.2500',
    },
    'MIC_TIO': {
        'top': '>8.0000',
        'bottom': '0.2500',
    },
    'MIC_GEN': {
        'top': '>=16.0000',
        'bottom': '<=0.2500',
    },
    'MIC_FIS': {
        'top': '>256.0000',
        'bottom': '<=16.0000',
    },
    'MIC_SXT': {
        'top': '>64.0000',
        'bottom': '<=0.1250',
    },
    'MIC_AZM': {
        'top': '>16.0000',
        'bottom': '<=1.0000',
    },
    'MIC_CHL': {
        'top': '>32.0000',
        'bottom': '<=2.0000',
    },
    'MIC_CIP': {
        'top': '>=4.0000',
        'bottom': '<=0.0150',
    },
    'MIC_NAL': {
        'top': '>32.0000',
        'bottom': '1.0000',
    },
    'MIC_TET': {
        'top': '>32.0000',
        'bottom': '<=4.0000',
    },
}
'''

def main(excel_filepath):
    """ Runs data processing scripts to turn MIC text values from Excel input data
        into class categories
    Args:
        excel_filepath: Metadata Excel file. MIC columns will have prefix 'MIC_'
    """
    logger = logging.getLogger(__name__)
    logger.info('MIC binning')

    # drugs = ["MIC_AMP", "MIC_AMC", "MIC_FOX", "MIC_CRO", "MIC_TIO", "MIC_GEN",
    #     "MIC_FIS", "MIC_SXT", "MIC_AZM", "MIC_CHL", "MIC_CIP", "MIC_NAL", "MIC_TET"]
    # convert = { key: lambda x: str(x) for key in drugs }
   
    # micsdf = pd.read_csv(excel_filepath, sep='\t', usecols=['run'] + drugs, skip_footer=1, skip_blank_lines=False, converters=convert)
    # micsdf = micsdf.set_index('run')

    micsdf = pd.read_excel(excel_filepath)
    micsdf = micsdf[["run","MIC_AMP", "MIC_AMC", "MIC_FOX", "MIC_CRO", "MIC_TIO", "MIC_GEN", "MIC_FIS", "MIC_SXT", "MIC_AZM", "MIC_CHL", "MIC_CIP", "MIC_NAL", "MIC_TET"]]
    micsdf = micsdf.set_index('run')

    classes = {}
    class_orders = {}
    for col in micsdf:
        logger.debug('Creating MIC panel for {}'.format(col))
        class_labels, class_order = bin(micsdf[col], col)

        class_order = consolidate_bins(class_order)

        drug = col.replace('MIC_', '')
        classes[drug] = pd.Series(class_labels, index=micsdf.index)
        class_orders[drug] = class_order

        logger.debug("Final MIC distribution:\n{}".format(classes[drug].value_counts()))

    c = pd.DataFrame(classes)

    if not os.path.exists('./amr_data'):
        os.mkdir('amr_data')
    cfile = "amr_data/mic_class_dataframe.pkl"#os.path.join(data_dir, 'interim', 'mic_class_dataframe.pkl')
    cofile = "amr_data/mic_class_order_dict.pkl"#os.path.join(data_dir, 'interim', 'mic_class_order_dict.pkl')
    joblib.dump(c, cfile)
    joblib.dump(class_orders, cofile)

def consolidate_bins(bins):
    #print("before",bins)
    index = 0
    while index < len(bins)-1:
        #a = int(''.join(filter(str.isdigit, bins[index])))
        #b = int(''.join(filter(str.isdigit, bins[index+1])))

        a = bins[index].split('=')[-1]
        a = bins[index].split('<')[-1]
        a = bins[index].split('>')[-1]

        b = bins[index+1].split('=')[-1]
        b = bins[index+1].split('<')[-1]
        b = bins[index+1].split('>')[-1]

        #print(a,b)

        if a==b:
            #bins = bins.delete(index)
            del bins[index]
            if index == 0:
                bins[index] = "<="+str(a)
            elif index == len(bins)-1:
                bins[index] = ">="+str(a)
        index+=1
    print('MIC values will be mapped to: ', bins)
    #print("after",bins)
    return bins

'''
    for col in micsdf:
        new_bins = []
        # Get the column index of the drug
        col_index = micsdf.columns.get_loc(col)
        # Set invalid entries to be NaN for easy deletion
        for index, row in micsdf.iterrows():
            x =row[col_index]
            x = str(x)
            if x !='nan' and x != '-': x = int(''.join(filter(str.isdigit, x)))
            if x !='nan' and x != '-' and x not in new_bins:
                new_bins.append(x)

        new_bins = np.asarray(new_bins)
        new_bins = np.sort(new_bins)

        print(col, new_bins)
'''

def bin(mic_series, drug):

    logger = logging.getLogger(__name__)

    panel = MICPanel()
    values, counts = panel.build(mic_series)

    logger.debug('Panel value frequency:')
    for m,c in zip(values, counts):
        logger.debug("{}, {}".format(m, c))


    # Initialize MIC bins
    panel.set_range(mic_ranges[drug]['top'], mic_ranges[drug]['bottom'])

    # Iterate through MIC values and assign class labels
    #logger.debug('MIC values will be mapped to: {}'.format(panel.class_labels))
    classes = []
    for m in mic_series:
        mgpl = MGPL(m)
        mlabel = str(mgpl)

        if mgpl.isna:
            classes.append(np.nan)
        else:
            if not mlabel in panel.class_mapping:
                raise Exception('Mapping error')
            else:
                classes.append(panel.class_mapping[mlabel])


    return(classes, panel.class_labels)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # Load environment
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())

    #main(snakemake.input[0])
    #print(snakemake.input[0])
    #print(sys.argv[1])
    main(sys.argv[1])
    #main('amr_data/no_ecoli_GenotypicAMR_Master.xlsx')