# CRISPR-MCA



# Source Code Documentation

* main.py - **Main entry for model files**
* Data - **Dataset processing and analysis files**
  * DataAnalysis
    * Imbalance analysis 
      * dataset.csv
      * Imbalabce.py
    * Mismatch analysis
      * all_mismatch_matrices.png
      * position_type_hotmap.py
    * Mismatch count
      * HeatMap.py
      * Mismatch Counts.png
  * DataEncoding - **Coding files for all models**
    * encodingList.py
  * DataExtension - **ESB Class Rebalancing Strategy**
    * genome
      * Genome-wide files
    * crisot.py
    * crisot_score_param.csv
    * Main.py
  * DataSets - **off-target datasets**
    * extension - **Expanded datasets and their characterization**
      * cleaned_Haeussler_Extension.csv
      * cleaned_Hek293t_Extension.csv
      * cleaned_K562Hek293_Extension.csv
      * cleaned_K562_Extension.csv
      * cleaned_Kleinstiver_Extension.csv
      * cleaned_Listgarten_Extension.csv
      * cleaned_SITE_Extension.csv
      * cleaned_Tasi_Extension.csv
      * Expansion Location Heat Map.py
      * Expansion number analysis.py
      * Imbalance analysis.py
      * Result
        * imbanlance.csv
        * output.csv
    * Indel - **Datasets containing indel and mismatches**
      * CIRCLE_seq.csv
      * GUIDE-Seq.csv
    * Mismatch  - **Mismatch-only datasets**
      * Haeussler.csv
      * Hek293t.csv
      * K562.csv
      * K562Hek293.csv
      * Kleinstiver.csv
      * Listgarten.csv
      * SITE.csv
      * Tasi.csv
* Models - **All model files**
  * ModelList.py
  * mymodel.py
* Result



# Key point

The genome-wide files used in the ESB class rebalancing strategy explained in DataExtension can be downloaded from the following address:
https://hgdownload2.soe.ucsc.edu/downloads.html



# Requestments

pandas
numpy
tensorflow==2.3.2
shap
matplotlib
scikit-learn
keras_multi_head
keras_layer_normalization
keras_bert
imblearn
seaborn
imbalanced-learn==0.11.0
imblearn==0.0
keras-layer-normalization==0.16.0
keras-multi-head==0.29.0
keras-pos-embd== 0.13.0
keras-position-wise-feed-forward==0.8.0
keras-transformer==0.40.0
tqdm==4.66.1
xlrd==2.0.1
zipp==3.15.0
