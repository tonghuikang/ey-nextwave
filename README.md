### Relevant kernels

Source dataset <br>
https://www.kaggle.com/huikang/ey-nextwave

Given dataset, pivot and produce numpy arrays representing the data <br>
https://www.kaggle.com/huikang/eyn-original 

Transforming numpy arrays into pandas dataframe pickle <br>
https://www.kaggle.com/huikang/eyn-original-df

Adding trivial features on top of the dataframe <br>
https://www.kaggle.com/huikang/eyn-pre-unravel-df

Definition of fold for standardisation across experiments <br>
https://www.kaggle.com/huikang/eyn-folds

Obtaining the full set of features for training embeddings <br>
https://www.kaggle.com/huikang/eyn-pre-unravel-full-targets-df

Dataset containing the embeddings generated <br>
https://www.kaggle.com/mrjerry/eynembedding

Finding the nearest last-seen-points and its target value <br>
https://www.kaggle.com/huikang/eyn-pre-cluster-unif

Taking the weight average of nearest last-seen-points <br>
https://www.kaggle.com/huikang/eyn-pre-cluster-calc

The actual model <br>
https://www.kaggle.com/huikang/eyn-df-lgbm-manyf


### Our solution in brief
- Transformation each "sequence" from the source dataset into an array of 21*13 elements.
- Order of trajectory is reversed. The last trajectory is always at position zero.
- 10 fold cross validation.
- Use of the best threshold from the training set.
- Embeddings as a feature
- LightGBM model with parameters crudely tuned.
- (Might be special in our solution) Cluster features.

### Benefits of our solution
- No additional information used
- Consistent documentation throughout (with shape of array printed)
- Fast training time and preprocessing time (except one step that could be faster)
- Visualisation of more important features
- Competitive score of 0.89441

### Could have done better
- Systematic documentation. Currently submission scores are recorded on kernels.
- Parameter search. Requires script writing and more extensive resources.
- Theory behind the F1 score and the optimal threshold.
- `eyn-original` takes 5 hours to run due to creation of numerous dataframes.
