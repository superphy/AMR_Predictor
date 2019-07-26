"""
This snakemake finds the top 11mers and searches for their locations.
If it is not found search intergenic,
If duplicates are found, expand to 15mer and narrow to top gene.
"""

import numpy as np
import pandas as pd

# how many kmers to look at
top_x = "5"

drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]

ids, = glob_wildcards("data/genomes/clean/{id}.fasta")
gids, = glob_wildcards("data/grdi_genomes/clean/{gid}.fasta")

rule all:
    input:
        expand("annotation/search/11mer_data/{dataset}_11mer_summary.csv", dataset = datasets)
rule find_top:
    input:
        "data/features/{drug}_1000feats_XGBtrainedOn{dataset}_testedOn{dataset}_fold1.npy"
    output:
        "annotation/search/11mer_data/{drug}_{dataset}_top"+top_x+".npy"
    shell:
        "python annotation/search/find_top_feats.py {input} {output} "+top_x

rule ncbi_prokka:
    input:
        "data/genomes/clean/{id}.fasta"
    output:
        "annotation/annotated_genomes/{id}/{id}.ffn",
    threads:
        1
    shell:
        "prokka {input} --outdir annotation/annotated_genomes/{wildcards.id} --prefix {wildcards.id} --cpus {threads} --force --compliant"

rule grdi_prokka:
    input:
        "data/grdi_genomes/clean/{gid}.fasta"
    output:
        "annotation/annotated_grdi_genomes/{gid}/{gid}.ffn",
    threads:
        1
    shell:
        "prokka {input} --outdir annotation/annotated_grdi_genomes/{wildcards.gid} --prefix {wildcards.gid} --cpus {threads} --force --compliant"

rule ncbi_single_line:
    input:
        "annotation/annotated_genomes/{id}/{id}.ffn"
    output:
        "annotation/annotated_genomes/{id}/{id}.ffns"
    shell:
        "awk \'!/^>/ {{ printf \"%s\", $0; n = \"\\n\" }} ; /^>/ {{ print n $0; n = \"\" }} ; END {{ printf \"%s\", n }}\' < {input}  > {output}"

rule grdi_single_line:
    input:
        "annotation/annotated_grdi_genomes/{gid}/{gid}.ffn"
    output:
        "annotation/annotated_grdi_genomes/{gid}/{gid}.ffns"
    shell:
        "awk \'!/^>/ {{ printf \"%s\", $0; n = \"\\n\" }} ; /^>/ {{ print n $0; n = \"\" }} ; END {{ printf \"%s\", n }}\' < {input}  > {output}"

rule ncbi_prokka_to_df:
    input:
        "annotation/annotated_genomes/{id}/{id}.ffns"
    output:
        "annotation/gffpandas_ncbi/{id}.pkl"
    run:
        shell("export OPENBLAS_NUM_THREADS=1")
        import pandas as pd
        import gffpandas.gffpandas as gffpd
        anno = gffpd.read_gff3("annotation/annotated_genomes/{0}/{0}.gff".format(wildcards.id))
        ### consider cleaning dataframe here ###
        anno.df.to_pickle(str(output))

rule grdi_prokka_to_df:
    input:
        "annotation/annotated_grdi_genomes/{gid}/{gid}.ffns"
    output:
        "annotation/gffpandas_grdi/{gid}.pkl"
    run:
        shell("export OPENBLAS_NUM_THREADS=1")
        import pandas as pd
        import gffpandas.gffpandas as gffpd
        anno = gffpd.read_gff3("annotation/annotated_grdi_genomes/{0}/{0}.gff".format(wildcards.gid))
        anno.df.to_pickle(str(output))

rule blast_feats:
    input:
        "annotation/search/11mer_data/{drug}_{dataset}_top"+top_x+".npy"
    output:
        "annotation/search/11mer_data/{dataset}_blast_hits/{drug}.pkl"
    run:
        if(wildcards.dataset == 'grdi' and wildcards.drug in ['FIS','AZM']):
            shell("touch {output}")
        else:
            alt_out = str(output).split('.')[0]+'.query'
            if not os.path.exists(os.path.abspath(os.path.curdir)+"/annotation/search/11mer_data/{}_blast_hits".format(wildcards.dataset)):
                os.mkdir(os.path.abspath(os.path.curdir)+"/annotation/search/11mer_data/{}_blast_hits".format(wildcards.dataset))
            top_feats = np.load("annotation/search/11mer_data/{}_{}_top{}.npy".format(wildcards.drug,wildcards.dataset,top_x))
            if(isinstance(top_feats[0],bytes)):
                top_feats = [i.decode('utf-8') for i in top_feats]
            assert(len(top_feats[0])==11)
            with open(alt_out,'a') as fh:
                for feat in top_feats:
                    fh.write(">{}\n".format(feat))
                    fh.write(feat+"\n")
            shell("blastn -task blastn-short -db data/master.db -query {alt_out} -ungapped -perc_identity 100 -dust no -word_size 11 -evalue 100000 -outfmt 6 -out {output}")

rule find_hits:
    input:
        a = expand("annotation/search/11mer_data/{dataset}_blast_hits/{drug}.pkl", drug = drugs, dataset = datasets),
        b = expand("annotation/gffpandas_ncbi/{id}.pkl", id = ids),
        c = expand("annotation/gffpandas_grdi/{gid}.pkl", gid = gids)
    output:
        "annotation/search/11mer_data/{dataset}_hits_for_{drug}.pkl"
    run:
        if(wildcards.dataset == 'grdi' and wildcards.drug == 'FIS'):
            shell("touch {output}")
        else:
            shell("python annotation/search/find_hits.py annotation/search/11mer_data/{wildcards.dataset}_blast_hits/{wildcards.drug}.pkl {wildcards.dataset} {wildcards.drug} {top_x}")

rule hit_summary:
    input:
        expand("annotation/search/11mer_data/{dataset}_hits_for_{drug}.pkl", dataset = datasets, drug = drugs)
    output:
        "annotation/search/11mer_data/{dataset}_11mer_summary.csv"
    shell:
        "python/search/hit_summary.py {wildcards.dataset} {output}"



"""
rule grep_feats:
    input:
        a = expand("annotation/search/11mer_data/{drug}_{dataset}_top"+top_x+".npy", drug = drugs, dataset = datasets),
        b = expand("annotation/annotated_genomes/{id}/{id}.ffns", id = ids),
        c = expand("annotation/annotated_grdi_genomes/{gid}/{gid}.ffns", gid = gids)
    params:
        dataset = "{dataset}",
        drug = "{drug}"
    output:
        "annotation/search/11mer_data/gene_hits_11mer_for_{drug}_{dataset}_1000feats.out"
    run:
        if(dataset == 'public'):
            annotation_location = "annotation/annotated_genomes/"
        else:
            annotation_location = "annotation/annotated_grdi_genomes/"
        for seq in np.load(input.a):
            for root, dirs, files in os.walk(annotation_location):
                for genome in dirs:
                    if(dataset == 'public'):
                        shell("echo '\n\nSearching_for_{seq}_in_{genome}_{params.drug}' >> {output} && grep -B 1 {seq} annotation/annotated_genomes/{genome}/{genome}.ffns >> {output} || echo 'not found' >> {output}")
                    else:
                        shell("echo '\n\nSearching_for_{seq}_in_{genome}_{params.drug}' >> {output} && grep -B 1 {seq} annotation/annotated_grdi_genomes/{genome}/{genome}.ffns >> {output} || echo 'not found' >> {output}")

rule summary:
    input:
        expand("annotation/search/gene_hits_11mer_for_{drug}_{dataset}_1000feats.out", drug = drugs, dataset = datasets)
    output:
        "annotation/search/11mer_direct_hits_{dataset}"
    shell:
        "python annotation/search/grep_results_to_pandas.py {wildcards.dataset}"
"""
