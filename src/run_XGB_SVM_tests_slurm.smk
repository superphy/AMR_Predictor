drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
rule all:
    input:
        expand("results/public1_{drug}", drug = drugs),
        expand("results/grdi2_{drug}", drug = drugs),
        expand("results/kh3_{drug}", drug = drugs),
        expand("results/public4_grdi_{drug}", drug = drugs),
        expand("results/grdi5_public_{drug}", drug = drugs),
        expand("results/grdi6_kh_{drug}", drug = drugs),
        expand("results/kh7_grdi_{drug}", drug = drugs)

    shell:
        "echo All tests deployed"

rule public1_drug:
    output:
        "results/public1_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x public -a {params.drug} -o {output} -f $i -m $j"; done; done'

rule grdi2_drug:
    output:
        "results/grdi2_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x grdi -a {params.drug} -o {output} -f $i -m $j"; done; done'

rule kh3_drug:
    output:
        "results/kh3_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x kh -a {params.drug} -o {output} -f $i -m $j"; done; done'

rule public4_grdi_drug:
    output:
        "results/public4_grdi_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x public -y grdi -a {params.drug} -o {output} -f $i -m $j"; done; done'

rule grdi5_public_drug:
    output:
        "results/grdi5_public_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x grdi -y public -a {params.drug} -o {output} -f $i -m $j"; done; done'

rule grdi6_kh_drug:
    output:
        "results/grdi6_kh_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x grdi -y kh -a {params.drug} -o {output} -f $i -m $j"; done; done'

rule kh7_grdi_drug:
    output:
        "results/kh7_grdi_{drug}"
    params:
        drug = "{drug}"
    shell:
        'export OMP_NUM_THREADS=16 && mkdir {output} && for j in SVM XGB; do for i in $(seq 100 100 3000 & seq 3500 500 10000); do sbatch -c 16 --mem 125G --wrap="python src/model.py -x kh -y grdi -a {params.drug} -o {output} -f $i -m $j"; done; done'
