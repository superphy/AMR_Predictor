# AMR_Predictor
Machine learning methods to predict the anti-microbial resistance of Salmonella.

## Setup
1. Clone repository
2. Install dependecies: run conda env create -f envi.yaml
3. Move genome fasta files in AMR_Predictor/genomes/raw
4. Start conda environment: run source activate skmer
5. Run snakemake -j X #Where X is the number of cores you want to run it on
   e.g.snakemake -j 64 


