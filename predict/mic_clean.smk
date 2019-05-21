
#################################################################

# Location of the MIC data
PUBLIC_MIC_SPREADSHEET_PATH = "predict/mic_labels.xlsx"

#################################################################

import pandas as pd
import numpy as np
import logging
import pickle
import yaml
import os,sys

sys.path.insert(0, os.path.abspath(os.path.curdir)+"/src/")
from mic import MICPanel

def transform(input, log, output,
    slice_cols=["run", "MIC_AMP", "MIC_AMC", "MIC_FOX", "MIC_CRO",
    "MIC_TIO", "MIC_GEN", "MIC_FIS", "MIC_SXT", "MIC_AZM",
    "MIC_CHL", "MIC_CIP", "MIC_NAL", "MIC_TET"]):

    logging.basicConfig(filename=log[0],level=logging.DEBUG)
    logging.info('MIC binning')

    micsdf = pd.read_excel(input[0])
    micsdf = micsdf[slice_cols]
    micsdf = micsdf.set_index(slice_cols[0])

    logging.debug('Using classes defined in {}'.format(input[1]))
    with open(input[1], 'r') as infh:
        mic_class_labels = yaml.load(infh)

        classes = {}
        class_orders = {}
        for col in micsdf:
            logging.debug('Creating MIC panel for {}'.format(col))
            drug = col.replace('MIC_', '')

            mic_series = micsdf[col]
            class_list = []
            panel = MICPanel()

            # Initialize MIC bins for this drug
            panel.build(mic_class_labels[drug])

            # Iterate through MIC values and assign class labels
            logging.debug('MIC values will be mapped to: {}'.format(panel.class_labels))

            i = 0
            for m in mic_series:
                i += 1
                mlabel, isna = panel.transform(m)

                if isna:
                    class_list.append(np.nan)
                else:
                    class_list.append(mlabel)

            logging.info("Invalid MICs found in dataset: {}".format(', '.join(panel.invalids)))

            classes[drug] = pd.Series(class_list, index=micsdf.index)
            class_orders[drug] = panel.class_labels

            logging.debug("Final MIC distribution:\n{}".format(classes[drug].value_counts()))

        c = pd.DataFrame(classes)

        cfile = output[0]
        cofile = output[1]
        c.to_pickle(cfile)
        pickle.dump(class_orders, open(cofile, "wb"))


rule all:
  input:
    "predict/genomes/public_mic_class_dataframe.pkl",
    "predict/genomes/public_mic_class_order_dict.pkl"


rule public:
    input:
        PUBLIC_MIC_SPREADSHEET_PATH,
        "data/class_ranges.yaml"
    log:
        "predict/logs/mic_public.log"
    output:
        "predict/genomes/public_mic_class_dataframe.pkl",
        "predict/genomes/public_mic_class_order_dict.pkl"
    run:
        transform(input, log, output)
