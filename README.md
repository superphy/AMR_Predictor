# AMR_Predictor
Machine learning methods to predict the anti-microbial resistance of Salmonella.

## If you would like to use prebuilt models against your own genomes
1. Clone repository (run `git clone https://github.com/superphy/AMR_Predictor.git`)
2. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
3. Install dependecies: run `conda env create -f data/envi.yaml`

If you do not have MIC labels and only want predictions, skip to step 6

4. Name your labels as mic_labels.xlsx and put them in predict/mic_labels.xlsx

In mic_labels.xlsc the names of the genomes need to be in a column titled run and the MIC values need to be in columns labeled  like MIC_AMP, MIC_AMC, etc

See data/no_ecoli_GenotypicAMR_Master.xlsx for acceptable MIC formats

5. Run `snakemake -s predict/mic_clean.smk`
6. Place genomes in predict/genomes/raw
7. Run `snakemake -j X -s predict/predict.smk` where X is the number of cores you wish to use
8. View results in predict/results.csv

## If you would like to run all of the tests
1. Clone repository (run `git clone https://github.com/superphy/AMR_Predictor.git`)
2. [Download anaconda or miniconda (python 3.7)](https://conda.io/miniconda.html (python 3.7)), instructions for that [are here](https://conda.io/docs/user-guide/install/index.html)
3. Install dependecies: run `conda env create -f data/envi.yaml`
4. Move public genomes into AMR_Predictor/data/genomes/raw
5. Move grdi genomes in AMR_Predictor/data/grdi_genomes/raw (optional, but remove grdi rules from Snakefile)
6. Start conda environment: run `source activate skmer`
7. Run the following command, where 'X' is the number of cores you wish to use

   `snakemake -j X`
8. Run the following command to run all of the tests

   `snakemake -s src/run_tests.smk` or 
   `snakemake -s src/run_XGB_SVM_tests_slurm.smk && hyperas.smk`(if using slurm) 
9. Run `all_data_figures.py all` to save all the results as figures

### Manually Running Tests
If you do not want to run all tests in step 8 above, run `src/model.py --help` to see how to run the model for a specific set of parameters.

