drugs=["AMP","AMC","AZM","CHL","CIP","CRO","FIS","FOX","GEN","NAL","SXT","TET","TIO"]
rule all:
    input:
        expand("results/public_{drug}", drug = drugs)
        expand("results/grdi_{drug}", drug = drugs),
        expand("results/kh_{drug}", drug = drugs),
        expand("results/public_grdi_{drug}", drug = drugs),
        expand("results/grdi_public_{drug}", drug = drugs),
        expand("results/grdi_kh_{drug}", drug = drugs),
        expand("results/kh_grdi_{drug}", drug = drugs)

    shell:
        "echo All tests deployed"

rule public_drug:
    output:
        "results/public_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x public -a {params.drug} -o {output} -f $i -m $j; done; done'

rule grdi_drug:
    output:
        "results/grdi_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x grdi -a {params.drug} -o {output} -f $i -m $j; done; done'

rule kh_drug:
    output:
        "results/kh_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kh -a {params.drug} -o {output} -f $i -m $j; done; done'

rule public_grdi_drug:
    output:
        "results/public_grdi_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x public -y grdi -a {params.drug} -o {output} -f $i -m $j; done; done'

rule grdi_public_drug:
    output:
        "results/grdi_public_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x grdi -y public -a {params.drug} -o {output} -f $i -m $j; done; done'

rule grdi_kh_drug:
    output:
        "results/grdi_kh_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x grdi -y kh -a {params.drug} -o {output} -f $i -m $j; done; done'

rule kh_grdi_drug:
    output:
        "results/kh_grdi_{drug}"
    params:
        drug = "{drug}"
    shell:
        'mkdir {output} && for j in SVM ANN XGB; do for i in $(seq 100 100 3000); do python src/model.py -x kh -y grdi -a {params.drug} -o {output} -f $i -m $j; done; done'
