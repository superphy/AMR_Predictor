### Abstract
_Salmonella enterica_ is a food-borne pathogen that causes gastrointestinal infections such as: typhoid fever, paratyphoid fever, and salmonellosis. In Canada, in 2016 alone, an estimated 87,500 people contracted Salmonella spp., resulting in 925 hospitalizations and 17 deaths (3). Whole genome sequencing (WGS) has now become a routine and cost effective method of analyzing genome sequences. These sequences can be used for a variety of applications such as: surveillance and outbreak response for bacteria that is harmful to human health. The growth of these bacterias can be inhibited with an antimicrobial agent, the amount of antimicrobial agent required is called the minimum inhibitory concentration (MIC). MIC values are currently determined using laboratory tests that are costly and time consuming (5). In order to reduce cost and time we aim to use machine learning models to predict MIC values from a WGS. Specifically, looking at serovars of S. enterica with a focus on determining the MIC of 13 different antimicrobials. The data used is from publicly available and Canadian WGS data sets generated as part of the Genomics Research and Development Initiative (GRDI) on AMR with their accompanying MIC values. By using two separate data sets of varying diversity, we hope to identify the genomic features responsible for differences in MIC, rather than genomic regions that are merely correlated with the differences. This will allow better models to be built that are capable of making better MIC predictions for these important serogroups in the future. Machine learning models will be used because they are fast, scalable, and only require WGS data and corresponding MIC values to be able to make a prediction. Successful completion of this project will lead to faster and more accurate MIC predictions based solely on genome sequence data. This could decrease the cost and reduce testing times of antimicrobial resistance in Salmonella enterica, leading to healthier Canadians.

### Introduction 

### Implementation
WGS data will be reformatted for use in the supervised machine learning models that will be used to predict MIC values.

 The raw WGS data will be converted to a numeric input based on the frequency of DNA sequence substrings of size "k", called k-mers. Counts of k-mers with length 11 will be computed for each genome using the program Jellyfish (6). A k-mer length of 11 was previously found to be optimal. Important k-mers with be catalogued and mapped back to specific genes to identify where in the genome the k-mer was found. This could help to identify  potentially unknown factors that contribute to AMR. 

We will use two machine learning methods: Gradient Boosted Decision Trees (XGBoost) (7) and artificial neural networks (ANN). XGBoost uses an ensemble of weaker decision trees that when working together to answer a series of true or false questions create a strong model to make a prediction. An ANN uses layers of neurons with corresponding weights to make a prediction based on specified input neurons. The XGBoost model will be used as implemented using the XGBoost Python package and the ANN using Keras (8), TensorFlow (9), and scikit-learn (10).

### Conclusion




### References 
1.      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5047448/
2.      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5622158/
3.      https://www.canada.ca/en/public-health/services/food-borne-illness-canada/yearly-food-borne-illness-estimates-canada.html
4.      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1082862/
5.      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4503790/
6.      https://github.com/gmarcais/Jellyfish
7.      https://xgboost.readthedocs.io/en/latest/
8.      https://keras.io/
9.      https://www.tensorflow.org/
10.     http://scikit-learn.org/stable/index.html
11.     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5765115/
