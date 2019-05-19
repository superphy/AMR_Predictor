#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys

if __name__ == '__main__':
    """
        This script takes in a text file in the format:
            Drug:AMC Genome:SRR1200748 Predicted:<=1.0000 Actual:2.0000 OBO:True Major?:NonMajor,
        formats it into a pandas dataframe and then saves it as csv.
        Run as genome_error_table_converter.py {filepath}
    """
    with open(sys.argv[1]) as file:
        counter = 0
        drug = 'AMC'
        index = 0
        error_library = []
        drug1 = []
        genomes1 = []
        pred1 = []
        act1 = []
        off_by_one1 = []
        major1 = []
        dup_count = 0
        for line in file:
            first_char = line[:1]
            if(first_char=='D' and line.count(' ')==5):
                #print(line)
                drug, genome, pred, act, off_by_one, major = line.split(' ')
                drug = drug.split(':')[1]
                genome = genome.split(':')[1]
                pred = pred.split(':')[1]
                act = act.split(':')[1]
                off_by_one = off_by_one.split(':')[1]
                major = major.split(':')[1]
                major = major.rstrip()

                #cleaning duplicates
                superkey = drug + genome
                if superkey in error_library:
                    dup_count +=1
                    print("{} duplicates removed".format(dup_count))
                else:
                    error_library = np.append(error_library, superkey)

                    #then add the contents of the line to the pandas buffer
                    drug1 = np.append(drug1, drug)
                    genomes1 = np.append(genomes1, genome)
                    pred1 = np.append(pred1, pred)
                    act1 = np.append(act1, act)
                    off_by_one1 = np.append(off_by_one1, off_by_one)
                    major1 = np.append(major1, major)
        df = pd.DataFrame()
        df['Drug'] = drug1
        df['Genome'] = genomes1
        df['Predicted'] = pred1
        df['Actual'] = act1
        df['OffByOne'] = off_by_one1
        df['MajorError?'] = major1

        outputfile = sys.argv[1].split('.')[0]
        df.to_csv(outputfile+'.csv')
