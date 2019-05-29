rule all:
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/kmer.snake")
        shell("snakemake -j {threads} -s src/mics.snakefile")
        shell("snakemake -j {threads} -s src/grdi_kmer.snake")
        shell("python src/amr_prep.py")
        shell("python src/amr_prep_grdi.py")
        shell("for i in AMP AMC FOX CRO TIO GEN FIS SXT AZM CHL CIP NAL TET; do python src/ken_hei.py $i; done")
        shell("for i in AMP AMC FOX CRO TIO GEN FIS SXT AZM CHL CIP NAL TET; do for j in public grdi_ kh_ ; do python src/remove_mic.py $i $j; done ; done" )
 
