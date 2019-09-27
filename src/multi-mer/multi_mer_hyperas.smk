#drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
drugs = 'AMC'
splits=["1","2","3","4","5"]
#feats=[i for i in range(100,3000,100)]+[i for i in range(3000, 10500, 500)]
feats = [1000000]
rule all:
    input:
        expand("results/public1_{drug}/{drug}_{feat}feats_ANNtrainedOnpublic_testedOnaCrossValidation.pkl", drug = drugs, feat = feats)

rule split:
    output:
        "data/filtered/{drug}_31mer/splits/set1/"
    shell:
        "sbatch -c 1 --mem 32G --wrap='python src/validation_split_hyperas.py public {wildcards.drug} 1000000 31'"

rule hyperas:
    input:
        expand("data/filtered/{drug}/splits/set1/", drug = drugs)
    output:
        "data/{drug}_31mer/hyperas/{feat}feats_{split}.pkl"

    shell:
        "sbatch -c 144 --mem 1007G --wrap='python src/hyp.py {wildcards.feat} {wildcards.drug} 10 {wildcards.split} public 31'"

rule average:
    input:
        expand("data/{drug}/hyperas/{feat}feats_{split}.pkl", drug = drugs, feat = feats, split = splits)
    output:
        "results/multi-mer/31mer/public1_{drug}/{drug}_{feat}feats_ANNtrainedOnpublic_testedOnaCrossValidation.pkl"
    shell:
        "sbatch -c 1 --mem 2GB --wrap='python src/hyp_average.py {wildcard.feat} {wildcard.drug} public 31'"
