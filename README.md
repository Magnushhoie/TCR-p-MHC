# TCR-p-MHC
TCR-peptide-MHC binding predictor, using protein-protein interaction CNN in Pytorch and fast.ai.


Based on master thesis "Deep learning using protein structural modeling for prediction of T-cell receptor and peptide MHC binding", June 2019, attended at Danish Technical University under supervisor Paolo Marcatili.

Notebook overview:

- 0_RandomForestAnalysis1.ipynb and Analysis2.ipynb: Random forest analysis of original dataset by Olsen
- 1_Raw_data_preprocessing.ipynb: All pre-processing done to generate data/03_Dataset/preproc_dataset.zip. Basis for all training of CNN.
- 2_Raw_data_graphs_heatmaps.ipynb: Heatmaps of the raw dataset distribution
- 3_Performance_score_graphs.ipynb: Data analysis based on model performances and predictions. Scores were extracted and graphed based on CSV logs.
- 4_Prediction_Graphs_Logoplots.ipynb: Data analysis based on model performances and predictions. Scores were extracted and graphed based on CSV logs.

Data overview:

- data/00_model_structures: Raw dataset. Original TCR-p-MHC structural models downloaded from IEDB.org, split into 5 separate partitions, clustered by 30 % similarity
- data/00_Data_analysis: Various files used for the data analysis in notebooks 2, 3 and 4
- data/02_Features: Various datasets used for data analysis in notebooks 0, 2, 3 and 4. Includes csv files for FoldX energy terms, identity scores, full aligned sequences, BLOSUM-alignment scores and more for all 1464 complexes.
- data/03_Dataset: Full pre-processed dataset for which all training and validation was done. All performance statistics were done using this pre-processed dataset, except the Random Forest analysis using BLOSUM-alignment scores
- CSV: CSV log files with predictions and performance statistics from several thousand independent model with different hyperparameter settings. Generally at least 100 models were trained for each hyperparameter setting.
- results: Prediction graphs and logoplots generated using notebooks 3 and 4


Requirements (as used in conda environment):
- Python 3.7
- PyTorch 1.0.0
- Fast.ai 1.0.42
  - Fast.ai 0.7 may be required for some RandomForest scripts https://forums.fast.ai/t/fastai-v0-7-install-issues-thread/24652
- Pandas 0.24.0
- Numpy 1.15.4
- scikit-learn 0.20.2
- Scipy 1.2.0
- sklearn-pandas 1.8.0
- Plotnine 0.5.1
- MPL logoplot
  - https://github.com/micked/mpl-logoplot
- BioPython (pairwise alignment scores)
- pdpbox (RF partial dependence plots)
  - https://github.com/SauceCat/PDPbox
- treeinterpreter 0.2.2 (RF)
  - https://github.com/andosa/treeinterpreter







