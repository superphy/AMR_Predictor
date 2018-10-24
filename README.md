# AMR_Predictor
Machine learning methods to predict the anti-microbial resistance of Salmonella.

## Setup
1. Clone repository
2. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
3. Install dependecies: run conda env create -f envi.yaml
4. Move training genomes into AMR_Predictor/genomes/raw
5. Move testing genomes in AMR_Predictor/grdi_genomes/raw
6. Start conda environment: run source activate skmer
7. Run the following command, where 'X' is the number of cores you wish to use

   `snakemake -j X`
8. Check results in AMR_Predictor/amr_data/{drug}/{feat}features/grdi_xgb_report.txt


