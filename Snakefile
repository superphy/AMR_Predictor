
rule all:
    input:
        "data/AMP/500features/grdi_xgb_report.txt"

rule m_public_kmers:
    output:
        "data/unfiltered/kmer_matrix.npy"
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/kmer.snake")
rule m_grdi_kmers:
    output:
        "data/grdi_unfiltered/kmer_matrix.npy"
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/grdi_kmer.snake")
rule m_mics:
    input:
        "data/unfiltered/kmer_matrix.npy",
        "data/grdi_unfiltered/kmer_matrix.npy"
    output:
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/mics.snakefile")
rule m_feature_selection:
    input:
        "data/public_mic_class_dataframe.pkl",
        "data/public_mic_class_order_dict.pkl",
        "data/grdi_mic_class_dataframe.pkl",
        "data/grdi_mic_class_order_dict.pkl"
    output:
        'data/AMP/500features/X_train.pkl'
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/features.snakefile")
rule m_model_maker:
    input:
        'data/AMP/500features/X_train.pkl'
    output:
        "data/AMP/500features/xgb_report.txt"
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/xgboost.snakefile")
rule m_predict:
    input:
        "data/AMP/500features/xgb_report.txt"
    output:
        "data/AMP/500features/grdi_xgb_report.txt"
    threads:
        4096
    run:
        shell("snakemake -j {threads} -s src/xgboost_grdi.snakefile")
