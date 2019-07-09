# TCR-p-MHC
TCR-peptide-MHC binding predictor, using protein-protein interaction CNN in Pytorch and fast.ai.


Based on master thesis "Deep learning using protein structural modeling for prediction of T-cell receptor and peptide MHC binding", June 2019, attended at Danish Technical University under supervisor Paolo Marcatili.

Data overview:

- data/00_model_structures: original TCR-p-MHC structural models downloaded from IEDB.org, split into 5 separate partitions, clustered by 30 % similarity
- data/00_Data_analysis: Various files used for the data analysis in notebooks 2, 3 and 4
- data/02_Features: Various datasets used for data analysis in notebooks 0, 2, 3 and 4. Includes csv files for FoldX energy terms, identity scores, full aligned sequences, BLOSUM-alignment scores and more for all 1464 complexes.
- data/03_Dataset: Full pre-processed dataset for which all training and validation was done. All performance statistics were done using this pre-processed dataset, except the Random Forest analysis using BLOSUM-alignment scores
- CSV: CSV log files with predictions and performance statistics from several thousand independent model with different hyperparameter settings. Generally at least 100 models were trained for each hyperparameter setting.
- results: Prediction graphs and logoplots generated using notebooks 3 and 4





