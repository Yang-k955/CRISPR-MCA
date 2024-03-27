CRISPR-MCA is a novel hybrid network model for off-target prediction in CRISPR-Cas9, which employs multi-scale feature extraction and fusion techniques to improve the prediction accuracy while ensuring the model is lightweight. We also open-source the Efficiency and Specificity-Based (ESB) class rebalancing strategy, which can solve the problem of class imbalance in off-target datasets.



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
