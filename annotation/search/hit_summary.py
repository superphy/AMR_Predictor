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

    all_hits = []

    for drug in drugs:
        if(dataset == 'grdi' and drug in ['FIS','AZM']):
            continue
        df_path = "annotation/search/11mer_data/{}_hits_for_{}.pkl".format(dataset,drug)
        drug_list = merge_df(df_path, drug, dataset)
        for hit in drug_list:
            all_hits.append(hit)

    data = np.asarray(all_hits)

    what_amg = np.zeros((data.shape[0]), dtype = object)

    amgs = pd.read_csv("data/gene_labels.tsv",sep='\t')

    for i, val in enumerate(what_amg):
        for j, amg_list in enumerate(amgs['gene_codes']):
            for amg in amg_list:
                if(data[i][3].split('_')[0] in amg):
                    what_amg[i] = amgs['AMR Gene Family'][j]

    all_df = pd.DataFrame(data=np.concatenate((data,np.asarray([what_amg]).T),axis=1),columns=['dataset','drug','kmer','gene','name','count','average_dist', 'std_dev' ,"AMG"])

    all_df.to_csv(output)
