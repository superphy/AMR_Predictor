rule all:
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/grdi_kmer.snake")
        shell("snakemake -j {threads} -s src/mics.snakefile")
        shell("python src/amr_prep_grdi.py")
        shell("python src/xgb_test_grdi.py 2000 AMP")
