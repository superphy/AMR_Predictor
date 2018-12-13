# AMR_Predictor
Machine learning methods to predict the anti-microbial resistance of Salmonella.

## Setup
1. Clone repository (run `git clone https://github.com/superphy/AMR_Predictor.git`)
2. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
3. Install dependecies: run `conda env create -f data/envi.yaml`
4. Move public genomes into AMR_Predictor/genomes/raw
5. Move grdi genomes in AMR_Predictor/grdi_genomes/raw (optional, but remove run from Snakefile)
6. Start conda environment: `run source activate skmer`
7. Run the following command, where 'X' is the number of cores you wish to use

   `snakemake -j X`
8. Run the following command to run all of the tests

   `snakemake -s src/run_tests.smk` or `snakemake -s src/slurm_run_tests.smk`(if using slurm) 
9. Run `for dir in results/*; do python src/figures.py $dir/; done` to save all the results as figures

### Manually Running Tests
If you do not want to run all tests in step 6 of setup, run `src/model.py --help` to see how to run the model for a specific set of parameters.

