# MVFormer
# 1. Description
MVFormer is a method to predict the pathogenicity of missense variants using gated transformers. This method first uses the tokenization-embedding paradigm in natural language processing to encode features of missense variants and convert features to token-level embeddings. Then TransSNP converts token-level embeddings to sample-level embeddings by a gated transformer model and uses a classifier to generate final pathogenicity scores. 
# 2. Train model
To train TransSNP for pathogenicity prediction, run the following command:  
`python ./train_model.py './data/clinvar_2022_missense' './data/exovar'`  
Note that the first data set in this command is used as the training set and the second data set as the test set
# 3. Test model
To evaluate the performance of TransSNP using other datasets, run the following command:  
`python ./test_model.py './data/clinvar_2022_missense' './data/cancer_discover'`  
`python ./test_model.py './data/clinvar_2022_missense' './data/swissvar'`
# 4. Custom Dataset
If you want to use your own datasets, you should put the preprocessed table into   
```\data\your_own_datasets\data_processed.csv```
# Acknowledgements
The code is inspired by TransTab (Wang Z, Sun J. Transtab: Learning transferable tabular transformers across tables. Advances in Neural Information Processing Systems, 2022, 35: 2902-2915.)
