ids, = glob_wildcards("data/genomes/raw/{id}.fasta")
gids, = glob_wildcards("data/grdi_genomes/raw/{gid}.fasta")

kmer_length = '31'
set_nums = [str(i) for i in range(1,10)]
split_nums = [str(i) for i in range(1,47)]
grdi_splits = [str(i) for i in range(47,59)]
drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
datasets=["public","grdi"]
dataset_paths = ['','grdi_']

rule all:
    input:
        "results/multi-mer/"+kmer_length+"mer/top_hit_with_card.csv"
rule clean:
    input:
        "data/genomes/raw/{id}.fasta"
    output:
        "data/genomes/clean/{id}.fasta"
    shell:
        "python src/clean.py {input} data/genomes/clean/"

rule grdi_clean:
    input:
        "data/grdi_genomes/raw/{gid}.fasta"
    output:
        "data/grdi_genomes/clean/{gid}.fasta"
    shell:
        "python src/clean.py {input} data/grdi_genomes/clean/"

rule count:
    input:
        "data/genomes/clean/{id}.fasta"
    output:
        temp("data/genomes/jellyfish_results"+kmer_length+"/{id}.jf")
    threads:
        2
    shell:
        "jellyfish count -C -m "+kmer_length+" -s 100M -t {threads} {input} -o {output}"

rule grdi_count:
    input:
        "data/grdi_genomes/clean/{gid}.fasta"
    output:
        temp("data/grdi_genomes/jellyfish_results"+kmer_length+"/{gid}.jf")
    threads:
        2
    shell:
        "jellyfish count -C -m "+kmer_length+" -s 100M -t {threads} {input} -o {output}"

rule dump:
    input:
        "data/genomes/jellyfish_results"+kmer_length+"/{id}.jf"
    output:
        "data/genomes/jellyfish_results"+kmer_length+"/{id}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule grdi_dump:
    input:
        "data/grdi_genomes/jellyfish_results"+kmer_length+"/{gid}.jf"
    output:
        "data/grdi_genomes/jellyfish_results"+kmer_length+"/{gid}.fa"
    shell:
        "jellyfish dump {input} > {output}"

rule union:
    # runs 5 times for ncbi, 4 for grdi
    input:
        expand("data/genomes/jellyfish_results"+kmer_length+"/{id}.fa", id=ids)
    output:
        "data/genomes/top_feats/all_"+kmer_length+"mers{set_num}.npy"
    shell:
        "python src/multi-mer/find_master.py "+kmer_length+" {output} {set_num}"

rule merge_masters:
    # runs once per dataset
    input:
        expand("data/genomes/top_feats/all_"+kmer_length+"mers{set_num}.npy", set_num = set_nums)
    output:
        "data/genomes/top_feats/{dataset_path}all_"+kmer_length+"mers.npy"
    shell:
        "python src/multi-mer/find_all_feats.py "+kmer_length+" {output} {dataset_path}"

rule matrix:
# runs 46 times for ncbi, 12 times for grdi
    input:
        expand("data/genomes/top_feats/{dataset_path}all_"+kmer_length+"mers.npy", dataset_path = dataset_paths)
    output:
        "data/multi-mer/splits/"+kmer_length+"mer_matrix{split}.npy"
    run:
        import os,sys
        import numpy as np

        if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/multi-mer/genome_names.npy"):
            genomes = ([files for r,d,files in os.walk("data/genomes/jellyfish_results{}/".format(kmer_length))][0])
            np.save("data/multi-mer/genome_names.npy", genomes)
        if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/multi-mer/genome_names.npy"):
            genomes = ([files for r,d,files in os.walk("data/grdi_genomes/jellyfish_results{}/".format(kmer_length))][0])
            np.save("data/multi-mer/grdi_genome_names.npy", genomes)


        shell("sbatch -c 16 --mem 700G --partition NMLResearch --wrap='python src/multi-mer/multi_mer_matrix.py {input} {wildcards.split} {kmer_length}'")

rule merge:
    input:
        expand("data/multi-mer/splits/"+kmer_length+"mer_matrix{split}.npy", split = split_nums)
    output:
        "data/multi-mer/"+kmer_length+"mer_matrix.npy",
        "data/multi-mer/"+kmer_length+"mer_rows.npy",
        "data/multi-mer/"+kmer_length+"mer_cols.npy"
    shell:
        "python src/multi-mer/matrix_merge.py {kmer_length} public"

rule grdi_merge:
    input:
        expand("data/multi-mer/splits/"+kmer_length+"mer_matrix{split}.npy", split = grdi_splits)
    output:
        "data/multi-mer/grdi_"+kmer_length+"mer_matrix.npy",
        "data/multi-mer/grdi_"+kmer_length+"mer_rows.npy",
        "data/multi-mer/grdi_"+kmer_length+"mer_cols.npy"
    shell:
        "python src/multi-mer/matrix_merge.py {kmer_length} grdi"

rule model:
    input:
        "data/multi-mer/grdi_"+kmer_length+"mer_matrix.npy",
        "data/multi-mer/"+kmer_length+"mer_matrix.npy"
    output:
        "data/multi-mer/feat_ranks/{dataset}_1000_{drug}_"+kmer_length+"mer_feature_ranks.npy"
    run:
        if wildcards.dataset == 'grdi' and wildcards.drug == 'FIS':
            shell("touch {output}")
        else:
            shell("sbatch -c 144 --mem 1007G --partition NMLResearch --wrap='python src/multi-mer/multi_mer_model.py {drug} 144 {dataset} 31 1000000 0'")

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

rule find_top:
    input:
        "data/multi-mer/feat_ranks/{dataset}_1000000_{drug}_"+kmer_length+"mer_feature_ranks.npy"
    output:
        "data/multi-mer/feat_ranks/{dataset}_1000000_{drug}_"+kmer_length+"mer_top5_feats.npy"
    run:
        # we can use the find_top_feats.py but thats conditional on non CV split models having non zero importance
        # more on this as the results of multi mer model unfold.
        import os,sys
        import numpy as np
        import math
        sys.path.append('annotation/search/')
        from find_top_feats import find_top_feats

        if(wildcards.dataset == 'grdi' and wildcards.drug in ['AZM','FIS']):
            shell("touch {output}")
        else:
            feat_imps = np.load(input[0], allow_pickle=True)

            if(isinstance(feat_imps[0,0],bytes)):
                raise Exception("Feature ranks are bytes")

            feat_imps = np.array([[str(i) for i in feat_imps[0]],[float(i) for i in feat_imps[1]],[float(i) for i in feat_imps[2]]],dtype='object')

            assert(isinstance(feat_imps[1,0],float))

            for i in feat_imps[1]:
                assert(not math.isnan(i))

            np.save(output[0], find_top_feats(feat_imps, output[0], 5))


rule blast_feats:
    input:
        "data/multi-mer/feat_ranks/{dataset}_1000000_{drug}_"+kmer_length+"mer_top5_feats.npy"
    output:
        "data/multi-mer/blast/1000000_{dataset}_"+kmer_length+"mer_blast_hits/{drug}.pkl"
    run:
        if(wildcards.dataset == 'grdi' and wildcards.drug in ['AZM','FIS']):
            shell("touch {output}")
        else:
            import numpy as np
            alt_out = str(output).split('.')[0]+'.query'
            if not os.path.exists(os.path.abspath(os.path.curdir)+"/data/multi-mer/blast/1000_{}_"+kmer_length+"mer_blast_hits".format(wildcards.dataset)):
                os.mkdir(os.path.abspath(os.path.curdir)+"/data/multi-mer/blast/1000_{}_"+kmer_length+"mer_blast_hits".format(wildcards.dataset))
            top_feats = np.load(input[0], allow_pickle=True)
            #if(isinstance(top_feats[0], list)):
            top_feats = np.array(top_feats)[:,0]
            if(isinstance(top_feats[0],bytes)):
                top_feats = [i.decode('utf-8') for i in top_feats]
            #print(top_feats[0], kmer_length)
            assert(len(top_feats[0])==int(kmer_length))
            with open(alt_out,'a') as fh:
                for feat in top_feats:
                    fh.write(">{}\n".format(feat))
                    fh.write(feat+"\n")
            shell("blastn -task blastn-short -db data/master.db -query {alt_out} -ungapped -perc_identity 100 -dust no -word_size {kmer_length} -max_target_seqs 50000 -evalue 100000 -outfmt 6 -out {output}")

rule find_hits:
    input:
        a = "data/multi-mer/blast/1000000_{dataset}_"+kmer_length+"mer_blast_hits/{drug}.pkl",
        b = expand("annotation/gffpandas_ncbi/{id}.pkl", id = ids),
        c = expand("annotation/gffpandas_grdi/{gid}.pkl", gid = gids)
    output:
        "data/multi-mer/blast/1000000_{dataset}_"+kmer_length+"mer_blast_hits/{drug}_hits.pkl"
    run:
        if(wildcards.dataset == 'grdi' and wildcards.drug in ['FIS','AZM']):
            shell("touch {output}")
        else:
            shell("python annotation/search/find_hits.py {input.a} {wildcards.dataset} {wildcards.drug} 5 1000000 "+kmer_length)

rule hit_summary:
    input:
        expand("data/multi-mer/blast/1000000_{dataset}_"+kmer_length+"mer_blast_hits/{drug}_hits.pkl", drug = drugs, dataset = datasets)
    output:
        "results/multi-mer/"+kmer_length+"mer/{dataset}_"+kmer_length+"mer_summary.csv"
    shell:
        "python annotation/search/hit_summary.py {wildcards.dataset} {output} 31"

rule score_summary:
    input:
        expand("results/multi-mer/"+kmer_length+"mer/{dataset}_"+kmer_length+"mer_summary.csv", dataset = datasets)
    output:
        "data/multi-mer/feat_ranks/{}mer_score_summary.csv".format(kmer_length)
    run:
        import pandas as pd
        import numpy as np
        row = []
        for drug in drugs:
            for dataset in datasets:
                if dataset == 'grdi' and drug in ['AZM','FIS']:
                    continue
                imp_path = "data/multi-mer/feat_ranks/{}_1000000_{}_{}mer_feature_ranks.npy".format(dataset,drug,kmer_length)
                imp_arr = np.load(imp_path, allow_pickle=True)

                feat_path = "data/multi-mer/feat_ranks/{}_1000000_{}_{}mer_top5_feats.npy".format(dataset,drug,kmer_length)
                top_feats = np.load(feat_path, allow_pickle=True)

                for kmer in top_feats[:,0]:
                    indx = np.where(imp_arr[0]==kmer)
                    scores = imp_arr[:,indx]
                    row.append([dataset,drug,kmer,scores[1],scores[2]])
        df = pd.DataFrame(data=row,columns=['Dataset','Antimicrobial',kmer_length+'mer','XGBoost Score','f_classif'])
        df.to_csv("data/multi-mer/feat_ranks/{}mer_score_summary.csv".format(kmer_length))

rule find_best_hit:
    input:
        "data/multi-mer/feat_ranks/{}mer_score_summary.csv".format(kmer_length)
    output:
        "results/multi-mer/"+kmer_length+"mer/best_hits.csv"
    shell:
        "python annotation/search/select_best_hit.py "+kmer_length

rule add_card_hits:
    input:
        "results/multi-mer/"+kmer_length+"mer/best_hits.csv"
    output:
        "results/multi-mer/"+kmer_length+"mer/top_hit_with_card.csv"
    shell:
        "python src/multi-mer/CARD_hits.py 31"
