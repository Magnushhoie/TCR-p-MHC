# Deep-learning T-cell receptor to peptide-MHC binding predictor
Deep-learning based T-cell receptor binding predictor. Convolutional neural network based on protein-protein interaction model by [Hashemifar et al 2018](https://academic.oup.com/bioinformatics/article/34/17/i802/5093239) (see poster). Input features are sequences of TCR, peptide and MHC cognate parts sourced from IEDB.org, and protein free energies extracted by FoldX. Protein structural models constructed using homology modelling in MODELLER, and is described in further detail in the thesis. CNN is coded in Pytorch and fast.ai deep learning library, see CNN.py.

Thesis title was ["Deep learning using protein structural modeling for prediction of T-cell receptor and peptide MHC binding"](https://github.com/Magnushhoie/TCR-p-MHC/blob/master/MagnusHoie_MasterThesis_Final.pdf), June 2019, attended at Danish Technical University under supervisor Paolo Marcatili.

Project is based on previous work by Olsen TH:
"Combining deep learning and structural modeling to predict T cell receptor specificity" - DTU Findit. https://findit.dtu.dk/en/catalog/2438676047


#### Poster, presented at KU Science/Novo Nordisk workshop on “Data Science in the Pharmaceutical Industry", Nov 15 2019:
<img src="https://github.com/Magnushhoie/TCR-p-MHC/raw/master/tcr_p_mhc_poster_2019.png">

#### Data overview:

- data/00_model_structures: Raw dataset. Original TCR-p-MHC structural models downloaded from IEDB.org, split into 5 separate partitions, clustered by 30 % similarity
- data/02_Features: CSV files for FoldX energy terms, identity scores, full aligned sequences, BLOSUM-alignment scores and more for all 1464 complexes.
- data/03_Dataset: Full pre-processed dataset for which all training and validation was done. All performance statistics were done using this pre-processed dataset, except the Random Forest analysis using BLOSUM-alignment scores
- Results: Prediction graphs and logoplots

#### Requirements (as used in conda environment):
- Python 3.7.2
- PyTorch 1.0.0
- Fast.ai 0.7
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

