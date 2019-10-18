
def find_locs(kmer, blast_df):
    import pandas as pd
    locs = []

    # filter blast results to just our kmer of interest
    kmer_df = blast_df[blast_df['qseqid'] == kmer]

    assert(kmer_df.shape[0]>0)

    # for every kmer hit in the genome, append the location
    for i in range(kmer_df.shape[0]):
        send = kmer_df['send'].values[i]
        sstart = kmer_df['sstart'].values[i]
        genome_id = kmer_df['sseqid'].values[i].split('_')[0]
        contig_name = kmer_df['sseqid'].values[i]
        locs.append([send,sstart,genome_id,contig_name])

    # locs is 2d list, each row is (start, stop, genome_id, contig_name)
    return locs

def find_gene(start, stop, genome_id, contig_name, prokka_loc):
    import pandas as pd
    import gffpandas.gffpandas as gffpd

    gene_up = ''
    dist_up = -1
    gene_down = ''
    dist_down = -1

    # prokka renames contigs but the numbers are consistent, so we need to pull the number
    if("NODE" in contig_name):
        orig_contig_name = contig_name.split('|')
        assert(len(orig_contig_name)==2)
        orig_contig_name = orig_contig_name[1]
        contig_num = orig_contig_name.split('_')[1]

    elif(contig_name.split('_')[0] == genome_id and len(contig_name.split('_'))==2):
        contig_num = contig_name.split('_')[1]

    # SRR5573065_SRR5573065.fasta|33_length=41292_depth=1.01x
    elif(contig_name.split('_')[0] == genome_id and len(contig_name.split('_')) in [4,5]):
        contig_num = contig_name.split('|')[1].split('_')[0]

    else:
        raise Exception("Unexpected contig name found: {}".format(contig_name))

    if(prokka_loc[-5:-1]=='ncbi'):
        gff_loc = "annotation/annotated_genomes/"
    else:
        gff_loc = "annotation/annotated_grdi_genomes/"

    # scan through contigs until the correct number is found, then keep the contig name
    with open("{0}{1}/{1}.gff".format(gff_loc,genome_id)) as file:
        for line in file:
            if("_{} ".format(contig_num) in line):
                prokka_contig = line.split(" ")[1]
                break

    if('prokka_contig' not in locals()):
        print("Contig number {2} and contig name {3} not located in {0}{1}/{1}.gff".format(gff_loc,genome_id, contig_num, contig_name))
        return [gene_up, dist_up, gene_down, dist_down]

    # columns are: ['seq_id','source','type','start','end','score','strand','phase','attributes']
    #with open(prokka_loc+genome_id+'.pkl', 'rb') as fh:
    #    gff_df = skbio.io.read(fh, format="blast+6",into=pd.DataFrame,default_columns=True)
    gff_df = pd.read_pickle(prokka_loc+genome_id+'.pkl')

    # keep only the contig the kmer was found in and only show coding sequences (Prodigal)
    contig_df = gff_df[gff_df['seq_id']==prokka_contig]
    contig_df = contig_df[contig_df['type']=='CDS']

    start = int(start)
    stop = int(stop)

    df_length = contig_df.values.shape[0]

    # find the nearest gene/genes
    # for every gene found by prokka, does it contain the kmer or is it near?
    for gene_num, gene_anno in enumerate(contig_df.values):
        gene_start = int(gene_anno[3])
        gene_stop = int(gene_anno[4])

        try:

            if(start > gene_stop):
                if(gene_num==df_length-1):
                    # we are after the last gene
                    gene_dict = dict(i.split('=') for i in gene_anno[8].split(';'))
                    dist_up = start - gene_stop
                    if(gene_dict['product']=='hypothetical protein'):
                        gene_up = "HypoProt:hypothetical protein"
                    else:
                        gene_up = gene_dict['gene']+':'+gene_dict['product']
                    break

                # we are not at the correct gene yet
                continue
            elif(stop < gene_start):
                # the kmer is before the current gene
                gene_dict = dict(i.split('=') for i in gene_anno[8].split(';'))
                dist_down = gene_start-stop
                if(gene_dict['product']=='hypothetical protein'):
                    gene_down = "HypoProt:hypothetical protein"
                else:
                    try:
                        gene_down = gene_dict['gene']+':'+gene_dict['product']
                    except KeyError:
                        gene_down = 'none:'+ dict(i.split('=') for i in gene_anno[8].split(';'))['product']

                prev_gene_anno = contig_df.values[gene_num-1]

                gene_dict = dict(i.split('=') for i in prev_gene_anno[8].split(';'))
                dist_up = start - prev_gene_anno[4]
                if(gene_dict['product']=='hypothetical protein'):
                    gene_up = "HypoProt:hypothetical protein"
                else:
                    gene_up = gene_dict['gene']+':'+gene_dict['product']
                break

            elif(start >= gene_start and stop <= gene_stop):
                # the kmer is inside of a gene
                gene_dict = dict(i.split('=') for i in gene_anno[8].split(';'))
                dist_up = 0
                if(gene_dict['product']=='hypothetical protein'):
                    gene_up = "HypoProt:hypothetical protein"
                else:
                    gene_up = gene_dict['gene']+':'+gene_dict['product']
                break

            elif(start <= gene_stop <= stop):
                # kmer hanging over right end of a gene
                gene_dict = dict(i.split('=') for i in gene_anno[8].split(';'))
                dist_up = stop-gene_stop
                if(gene_dict['product']=='hypothetical protein'):
                    gene_up = "HypoProt:hypothetical protein"
                else:
                    gene_up = gene_dict['gene']+':'+gene_dict['product']
                break

            elif(start <= gene_start <= stop):
                # kmer hanging over the left end of a gene
                gene_dict = dict(i.split('=') for i in gene_anno[8].split(';'))
                dist_up = gene_start-start
                if(gene_dict['product']=='hypothetical protein'):
                    gene_up = "HypoProt:hypothetical protein"
                else:
                    gene_up = gene_dict['gene']+':'+gene_dict['product']
                break

            else:
                raise Exception("Unexpected kmer start: {} stop: {} in relation to gene start: {} stop: {}".format(start, stop, gene_start, gene_stop))
        except KeyError:
            gene_up = 'none:'+ dict(i.split('=') for i in gene_anno[8].split(';'))['product']
            break

    return [gene_up, dist_up, gene_down, dist_down]

if __name__ == "__main__":
    import os,sys
    import skbio.io
    import pandas as pd
    import numpy as np

    blast_path = sys.argv[1]
    dataset = sys.argv[2]
    drug = sys.argv[3]
    top_x = sys.argv[4]
    num_feats = sys.argv[5]
    kmer_length = sys.argv[6]

    if(dataset == 'grdi' and drug in ['FIS','AZM']):
        raise Exception("Called find_hits.py for FIS using GRDI dataset, the snakemake shouldnt allow this")

    if kmer_length == '11':
        top_feats = np.load("annotation/search/11mer_data/{}_{}_top{}.npy".format(drug, dataset, top_x), allow_pickle = True)
    else:
        top_feats = np.load("data/multi-mer/feat_ranks/{}_{}_{}_{}mer_top5_feats.npy".format(dataset, num_feats, drug, kmer_length), allow_pickle = True)

    with open(blast_path) as fh:
        blast_df = skbio.io.read(fh, format="blast+6",into=pd.DataFrame,default_columns=True)

    if(dataset == 'public'):
        prokka_loc = "annotation/gffpandas_ncbi/"
    else:
        prokka_loc = "annotation/gffpandas_grdi/"

    # each row in gene hits will be [kmer, gene_up, dist_up, gene_down, dist_down, start, stop, genome_id, contig_name]
    gene_hits = []
    for kmer in top_feats:
        if kmer_length != '11':
            kmer = kmer[0]
        # locs is 2d list, each row is (start, stop, genome_id, contig_name)
        locs = find_locs(kmer, blast_df)
        for loc in locs:
            # ignore SA genome hits in public and SRR hits in grdi
            if((loc[2][:2] == 'SA' and dataset == 'public') or (loc[2][:3] == 'SRR' and dataset=='grdi')):
                continue
            # gene_info is 1D list: gene_up, dist_up, gene_down, dist_down
            gene_info = find_gene(*loc, prokka_loc)
            gene_hits.append([kmer]+gene_info+loc)

    # save gene hits as a dataframe
    hits_df = pd.DataFrame(data = np.asarray(gene_hits),columns = ['kmer', 'gene_up', 'dist_up', 'gene_down', 'dist_down', 'start', 'stop', 'genome_id', 'contig_name'])
    if(kmer_length == '11'):
        hits_df.to_pickle("annotation/search/11mer_data/{}_hits_for_{}.pkl".format(dataset,drug))
    else:
        hits_df.to_pickle("data/multi-mer/blast/{}_{}_{}mer_blast_hits/{}_hits.pkl".format(num_feats, dataset, kmer_length, drug))
