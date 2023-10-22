### Incremental Machine Learning

This project demonstrates the process of
- Preprocessing and incrementally training SGD classifer on a dataset having ~33.5M samples and 23 features (~9.5 GB in size).
- Parallelizing the EDA and data preprocessing tasks using multiprocessing package. 

Following are the steps involved in the project
1. Split data into train, validation and test sets.
2. EDA
   - Find data shape.
   - Find data types.
   - Find min, max and mean of numeric columns.
   - Find value counts of categorical columns.
   - Find misisng value couts and proportion.
   - Find target distribution.
3. Data preprocessing
   - Clean Size and Install columns.
   - Clean date columns and compute the no. of days elapsed till date.
   - Delete unwanted columns.
   - Label encoding and compute class weights.
   - Missing value Imputation.
   - Rare category encoding.
   - Boolean feature encoding.
   - One hot encoding.
   - Dimensionality reduction using IncrementalPCA.
4 Incremental model training using SGDClassifier.
5 Model evaluation using AUC ROC on train, validation and tests sets.

For this project, the <a href="https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps">Google Playstore dataset</a> downloaded from Kaggle (~2.3M samples and ~676 MB in size) is replicated multiple times to get ~50M samples (~14 GB in size). The dataset is split into training set (~33.5M samples, ~9.5 GB in size), validation set (~8M samples, ~2 GB in size) and test set (~8M samples, ~2 GB in size).
