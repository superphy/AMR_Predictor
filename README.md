# AMR_Predictor
Machine learning methods to predict the anti-microbial resistance of Salmonella.

## Setup
1. Clone repository
2. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
3. Install dependecies: run conda env create -f envi.yaml
4. Move genome fasta files in AMR_Predictor/genomes/raw
5. Start conda environment: run source activate skmer
6. Run snakemake -j X #Where X is the number of cores you want to run it on
   e.g.snakemake -j 64 
7. Run the test of your choosing, e.g. xgb.snake

Note that most tests will be hyperparameter optimized and may take some time to run.

