
def find_locs(kmer, blast_df):
    import pandas as pd
    locs = []

    # filter blast results to just our kmer of interest
    kmer_df = blast_df[blast_df['qseqid'] == kmer]

    assert(kmer_df.shape[0]>0)

    # for every kmer hit in the genome, append the location
    for i in range(kmer_df.shape[0]):
        locs.append([kmer_df['sstart'][i],kmer_df['send'][i],kmer_df['sseqid'][i].split('_')[0],kmer_df['sseqid'][i]])
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

    elif(contig_name.split('_')[0] == genome_id and len(contig_name.split('_')==2))
        contig_num = contig_name.split('_')[1]

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

    # columns are: ['seq_id','source','type','start','end','score','strand','phase','attributes']
    with open(prokka_loc+genome_id+'.pkl') as fh:
        gff_df = skbio.io.read(fh, format="blast+6",into=pd.DataFrame,default_columns=True)

    # keep only the contig the kmer was found in and only show coding sequences (Prodigal)
    contig_df = gff_df[gff_df[seq_id]==prokka_contig]
    contig_df = contig_df[contig_df['type']=='CDS']

    start = int(start)
    stop = int(stop)

    df_length = contig_df.values.shape[0]

    # find the nearest gene/genes
    # for every gene found by prokka, does it contain the kmer or is it near?
    for gene_num, gene_anno in enumerate(contig_df.values):
        gene_start = int(gene_anno[3])
        gene_stop = int(gene_anno[4])

        if(start > gene_stop):
            if(gene_num==df_length-1):
                # we are after the last gene
                gene_up = dict(i.split('=') for i in gene_anno[8].split(';'))['product']
                dist_up = start - gene_stop
                break

            # we are not at the correct gene yet
            continue
        elif(stop < gene_start):
            # the kmer is before the current gene
            gene_down = dict(i.split('=') for i in gene_anno[8].split(';'))['product']
            dist_down = gene_start-stop

            prev_gene_anno = contig_df.values[gene_num-1]

            gene_up = dict(i.split('=') for i in prev_gene_anno[8].split(';'))['product']
            dist_up = start - prev_gene_anno[4]
            break

        elif(start >= gene_start and stop <= gene_stop):
            # the kmer is inside of a gene
            gene_up = dict(i.split('=') for i in gene_anno[8].split(';'))['product']
            dist_up = 0
            break

        elif(start < gene_stop < stop):
            # kmer hanging over right end of a gene
            gene_up = dict(i.split('=') for i in gene_anno[8].split(';'))['product']
            dist_up = stop-gene_stop
            break

        elif(start < gene_start < stop):
            # kmer hanging over the left end of a gene
            gene_up = dict(i.split('=') for i in gene_anno[8].split(';'))['product']
            dist_up = gene_start-start
            break

        else:
            raise Exception("Unexpected kmer start: {} stop: {} in relation to gene start: {} stop: {}".format(start, stop, gene_start, gene_stop))

    return [gene_up, dist_up, gene_down, dist_down]

if __name__ == "__main__":
    import os,sys
    import skbio.io
    import pandas as pd

    blast_path = sys.argv[1]
    dataset = sys.argv[2]
    drug = sys.argv[3]
    top_x = sys.arv[4]

    if(dataset == 'grdi' and drug = 'FIS'):
        raise Exception("Called find_hits.py for FIS using GRDI dataset, the snakemake shouldnt allow this")
    top_feats = "annotation/search/11mer_data/{}_{}_top{}.npy".format(drug, dataset, top_x)

    with open(blast_path) as fh:
        blast_df = skbio.io.read(fh, format="blast+6",into=pd.DataFrame,default_columns=True)

    if(dataset == 'public'):
        prokka_loc = "annotation/gffpandas_ncbi/"
    else:
        prokka_loc = "annotation/gffpandas_grdi/"

    # each row in gene hits will be [kmer, gene_up, dist_up, gene_down, dist_down, start, stop, genome_id, contig_name]
    gene_hits = []

    for kmer in top_feats:
        # locs is 2d list, each row is (start, stop, genome_id, contig_name)
        locs = find_locs(kmer, blast_df)
            for loc in locs:
                # gene_info is 1D list: gene_up, dist_up, gene_down, dist_down
                gene_info = find_gene(*loc, prokka_loc)
                gene_hits.append([kmer]+gene_info+loc)

    # save gene hits as a dataframe
    hits_df = pd.DataFrame(data = np.asarray(gene_hits),columns = [kmer, gene_up, dist_up, gene_down, dist_down, start, stop, genome_id, contig_name])
    hits_df.to_pickle("annotation/search/11mer_data/{}_hits_for_{}.pkl".format(dataset,drug))
