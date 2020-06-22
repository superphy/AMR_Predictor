"""
Takes as input a hit_df and groups it for readability in a csv
"""
def merge_df(df_path, drug, dataset):
    from statistics import stdev
    """
    Takes path to pandas df with the columns:
    [kmer, gene_up, dist_up, gene_down, dist_down, start, stop, genome_id, contig_name]
    and returns a df with the columns:
    [drug, dataset, kmer, gene_up, gene_up_count, avg_dist_up, gene_down, gene_down_count, avg_dist_down]
    """
    print(df_path)
    df = pd.read_pickle(df_path)
    hit_summary = []

    for kmer in set(df['kmer']):
        # filter for only a single kmer
        kmer_df = df[df['kmer']==kmer]

        for gene_direc, gene_dist in [['gene_up','dist_up'],['gene_down','dist_down']]:
            for gene in set(kmer_df[gene_direc]):
                if(len(gene)==0):
                    continue
                # filter for only a single gene
                gene_df = kmer_df[kmer_df[gene_direc]==gene]

                total_dist = 0

                for dist in gene_df[gene_dist]:
                        total_dist += abs(float(dist))

                count = gene_df.shape[0]
                average_dist = total_dist/count
                if(len(gene_df[gene_dist])<2):
                    std_dev = 0
                else:
                    std_dev  = stdev([abs(int(float(i))) for i in gene_df[gene_dist]])

                try:
                    gene, name = gene.split(':')
                except:
                    print("Gene: {}".format(gene))
                    print("{} {}".format(drug,dataset))
                    gene, carb, name = gene.split(':')

                hit_summary.append([dataset, drug, kmer, gene, name, count, average_dist, std_dev])

    return hit_summary



if __name__ == "__main__":
    import os, sys
    import pandas as pd
    import numpy as np

    drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]


    dataset = sys.argv[1]
    output = sys.argv[2]
    kmer_length = sys.argv[3]

    all_hits = []

    for drug in drugs:
        if(dataset == 'grdi' and drug in ['FIS']):
            continue
        if(kmer_length == '11'):
            df_path = "annotation/search/11mer_data/{}_hits_for_{}.pkl".format(dataset,drug)
        else:
            df_path = "data/multi-mer/blast/1000000_{}_{}mer_blast_hits/{}_hits.pkl".format(dataset,kmer_length,drug)
        drug_list = merge_df(df_path, drug, dataset)
        for hit in drug_list:
            all_hits.append(hit)

    data = np.asarray(all_hits)

    #what_amg = np.zeros((2,data.shape[0]), dtype = object)
    what_amg = np.zeros((data.shape[0]), dtype = object)
    #print("what_amg1", what_amg.shape)
    print("what_amg2", what_amg.T.shape)
    print("what_amg3", np.asarray([what_amg]).T.shape)
    print("data", data.shape)

    amgs = pd.read_pickle("data/gene_labels.tsv")

    for i, val in enumerate(what_amg):
        for j, amg_list in enumerate(amgs['gene_codes']):
            for amg in amg_list:
                if(data[i][3].split('_')[0] in amg):
                    what_amg[i] = amgs['AMR Gene Family'][j]+' ({})'.format(amgs['Resistance Mechanism'][j])
                    #what_amg[0][i] = amgs['AMR Gene Family'][j]
                    #what_amg[1][i] = amgs['Resistance Mechanism'][j]

    # check if what_amg is looking correct
    #print(dataset)
    #print(what_amg)
    #sys.exit()

    all_df = pd.DataFrame(data=np.concatenate((data,np.asarray([what_amg]).T),axis=1),columns=['dataset','drug','kmer','gene','name','count','average_dist', 'std_dev' ,"AMG"])
    print(all_df)
    all_df.to_csv(output)
