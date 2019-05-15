drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
splits=["1","2","3","4","5"]
feats=[i for i in range(100,3000,100)]+[i for i in range(3000, 10500, 500)]
rule all:
    input:
        expand("results/public1_{drug}/{drug}_{feat}feats_ANNtrainedOnpublic_testedOnaCrossValidation.pkl", drug = drugs, feat = feats)

rule split:
    output:
        "data/filtered/{drug}/splits/set1/"
    params:
        drug  = "{drug}"
    shell:
        "sbatch -c 1 --mem 32G --wrap='python src/validation_split_hyperas.py public {params.drug}'"

rule hyperas:
    input:
        expand("data/filtered/{drug}/splits/set1/", drug = drugs)
    output:
        "data/{drug}/hyperas/{feat}feats_{split}.pkl"
    params:
        drug = "{drug}",
        split = "{split}",
        feat = "{feat}"
    shell:
        "sbatch -c 16 --mem 125G --wrap='python src/hyp.py {params.feat} {params.drug} 10 {params.split} public'"

rule average:
    input:
        expand("data/{drug}/hyperas/{feat}feats_{split}.pkl", drug = drugs, feat = feats, split = splits)
    output:
        "results/public1_{drug}/{drug}_{feat}feats_ANNtrainedOnpublic_testedOnaCrossValidation.pkl"
    params:
        drug = "{drug}",
        feat = "{feat}"
    shell:
        "sbatch -c 1 --mem 2GB --wrap='python src/hyp_average.py {params.feat} {params.drug} public'"
