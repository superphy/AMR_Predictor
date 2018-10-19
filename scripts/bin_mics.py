<<<<<<< HEAD
#!/usr/bin/env python
"""bin_mics.py
Convert MIC values into distinct classes. Save processed
classes as pandas dataframes.
"""

=======
>>>>>>> master
import os
import logging
import re
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


def main(excel_filepath, class_label_filepath):
    """ Runs data processing scripts to turn MIC text values from Excel input data
        into class categories

<<<<<<< HEAD
            class labels are defined, observed MIC classes in data are mapped to these classes. An error
            is thrown if no mapped classes is found. Provde a class_label_filepath.
=======
        There are two modes
            1. Only max and min are defined, all other observed MIC classes are assigned a bin provided
            they are not incompatible (i.e. >8 in a panel where the MIC values range from 1 - 32)
            2. All class labels are defined, observed MIC classes in data are mapped to these classes. An error
            is thrown if no mapped classes is found. To use this version, provde a class_label_filepath above.
>>>>>>> master
            The class_label file will outline all classes in YAML format:


                AMP:
                    - 0.5
                    - 1.0
                    - 2.0
                    - ...

                - or -

                AMP: [0.5, 1.0, 2.0, ...]


    Args:
        excel_filepath: Metadata Excel file. MIC columns will have prefix 'MIC_'
<<<<<<< HEAD
        class_label_filepath: MIC classes in yaml format
=======
        class_label_filepath: If this is provided, the class labels will be loaded, instead of inferred from data

>>>>>>> master
    """

    logger = logging.getLogger(__name__)
    logger.info('MIC binning')

    #micsdf = pd.read_excel(excel_filepath)
    micsdf = pd.read_csv(excel_filepath, sep='\t')

    micsdf = micsdf[
        ["run", "MIC_AMP", "MIC_AMC", "MIC_FOX", "MIC_CRO", "MIC_TIO", "MIC_GEN", "MIC_FIS", "MIC_SXT", "MIC_AZM",
         "MIC_CHL", "MIC_CIP", "MIC_NAL", "MIC_TET"]]

    micsdf = micsdf.set_index('run')

    # micsdf = micsdf[
    #     ["SANumber", "MIC_AMP", "MIC_AMC"]]

    mic_class_labels = None
    if class_label_filepath:
        logger.debug('Using classes defined in {}'.format(class_label_filepath))
        with open(class_label_filepath, 'r') as infh:
            mic_class_labels = yaml.load(infh)

    classes = {}
    class_orders = {}
    for col in micsdf:
        logger.debug('Creating MIC panel for {}'.format(col))
        drug = col.replace('MIC_', '')
        if mic_class_labels:
            class_labels, class_order = transform(micsdf[col], mic_class_labels[drug])
        else:
            class_labels, class_order = bin(micsdf[col], col)

        classes[drug] = pd.Series(class_labels, index=micsdf.index)
        class_orders[drug] = class_order

        logger.debug("Final MIC distribution:\n{}".format(classes[drug].value_counts()))

    c = pd.DataFrame(classes)

    
    cfile  = os.path.abspath(os.path.curdir)+"/amr_data/mic_class_dataframe.pkl"#os.path.join(data_dir, 'interim', 'mic_class_dataframe.pkl')
    cofile = os.path.abspath(os.path.curdir)+"/amr_data/mic_class_order_dict.pkl"#os.path.join(data_dir, 'interim', 'mic_class_order_dict.pkl')
    #cfile = snakemake.output[0]
    #cofile = snakemake.output[1]
    joblib.dump(c, cfile)
    joblib.dump(class_orders, cofile)



def bin(mic_series, drug):

    logger = logging.getLogger(__name__)

    panel = MICPanel()
    values, counts = panel.build(mic_series)

    logger.debug('Panel value frequency:')
    for m,c in zip(values, counts):
        logger.debug("{}, {}".format(m, c))


    # Initialize MIC bins

    panel.set_range(mic_ranges[drug]['top'], mic_ranges[drug]['bottom'])
    logger.debug('Top MIC range: {}'.format(mic_ranges[drug]['top']))
    logger.debug('Bottom MIC range: {}'.format(mic_ranges[drug]['bottom']))
    # Iterate through MIC values and assign class labels
    logger.debug('MIC values will be mapped to: {}'.format(panel.class_labels))
    classes = []
    for m in mic_series:
        mgpl = MGPL(m)
        mlabel = str(mgpl)

        if mgpl.isna:
            classes.append(np.nan)
        else:
            if not mlabel in panel.class_mapping:
                raise Exception('Mapping error: class {} not found in MIC panel.'.format(mlabel))
            else:
                classes.append(panel.class_mapping[mlabel])


    return(classes, panel.class_labels)


def transform(mic_series, class_labels):

    logger = logging.getLogger(__name__)

    panel = MICPanel()

    # Initialize MIC bins
    panel.build(class_labels)

    # Iterate through MIC values and assign class labels
    logger.debug('MIC values will be mapped to: {}'.format(panel.class_labels))
    classes = []
    i = 0
    for m in mic_series:
        i += 1
        mlabel, isna = panel.transform(m)

        if isna:
            classes.append(np.nan)
        else:
            classes.append(mlabel)

    logger.info("Invalid MICs found in dataset: {}".format(', '.join(panel.invalids)))

    return (classes, panel.class_labels)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # Load environment
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    load_dotenv(find_dotenv())

    # Check for class file
    clfp = None
    #if hasattr(snakemake.params, 'class_labels') and snakemake.params.class_labels:
        #clfp = snakemake.params.class_labels

    main(sys.argv[1], clfp)
