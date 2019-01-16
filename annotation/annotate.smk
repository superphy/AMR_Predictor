"""
Note, in order for this snakemake to work you need to have run the test
that used the features that we are not mapping to the genome
"""

GENOMES_PATH = "data/genomes/clean/"

# How many features to map back to the genome
top_x_feats = '5'

# What drug to search for important kmers in
drug = 'AMP'

# Number of features the model was train on
feats = '3000'

# Training set
train ='public'

# Testing set
test = 'cv'

# fold
fold = '1'


ids, = glob_wildcards(GENOMES_PATH+"{id}.fasta")

rule all:
    input:
        'annotation/feature_summary_for_'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.out'

rule prokka:
    input:
        GENOMES_PATH+"{id}.fasta"
    output:
        "/annotation/annotated_genomes/{id}/{id}.ffn",
    threads:
        7
    shell:
        "prokka {input} --outdir annotation/annotated_genomes/{wildcards.id} --prefix {wildcards.id} --cpus {threads} --force"

rule single_line:
    input:
        "annotation/annotated_genomes/{id}/{id}.ffn"
    output:
        "annotation/annotated_genomes/{id}/{id}.ffns"
    shell:
        "awk \'!/^>/ {{ printf \"%s\", $0; n = \"\\n\" }} ; /^>/ {{ print n $0; n = \"\" }} ; END {{ printf \"%s\", n }}\' < {input}  > {output}"

rule grep_feats:
    input:
        a = expand("annotation/annotated_genomes/{id}/{id}.ffns", id = ids),
        b = 'data/features/'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.npy'
    output:
        'annotation/feature_summary_for_'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.out'
    run:
        import numpy as np
        top_feats = []
        all_feats = np.load('data/features/'+drug+'_'+feats+'feats_XGBtrainedOn'+train+'_testedOn'+test+'_fold'+fold+'.npy')
        global top_x_feats
        top_x_feats = int(top_x_feats)
        while(top_x_feats>0):
            m = max(all_feats[1])
            top_indeces = [i for i, j in enumerate(all_feats[1]) if j == m]
            top_x_feats -= len(top_indeces)
            for i in top_indeces:
                top_feats.append((all_feats[0][i]).decode('utf-8'))
                all_feats[1][i] = 0
        #shell("echo {top_feats} && touch {output}")
        for seq in top_feats:
            for root, dirs, files in os.walk("annotation/annotated_genomes/"):
                for genome in dirs:
                    shell("grep -B 1 {seq} annotation/annotated_genomes/{genome}/{genome}.ffns >> {output}")
