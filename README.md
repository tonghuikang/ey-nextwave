# Ernst & Young NextWave Data Science Competition
https://datascience.ey.com/challenge/1

## Results
We are 6th globally and 3rd in Singapore, with a public F1 score of 0.89441 and private F1 score of 0.88923.
- The kernel of the final submission: https://www.kaggle.com/huikang/eyn-df-lgbm-manyf?scriptVersionId=13913268

## Challenge
The dataset is a set of anonymized geolocation data of multiple mobile devices in the City of Atlanta (US) for 11 working days in October 2018. In each "device", there may be up to 20 "trajectories". Each trajectory has the following information.

| Variable name | Type    | Description 
| --------------|---------|-------------
| hash          | String  | Represents the unique identifier of a device
| trajectory_id | String  | Represents the unique identifier of a trajectory associated to a device
| time_entry    | Date    | Indicates the local time for the starting point of the trajectory (HH:mm:ss)
| time_exit     | Date    | Indicates the local time for the ending point of the trajectory (HH:mm:ss)
| Vmax          | Integer | Represents the maximum velocity registered in the course of a trajectory.
| Vmin          | Integer | Represents the minimum velocity registered in the course of a trajectory.
| Vmean         | Integer | Represents the average velocity registered in the course of a trajectory.
| x_entry       | Double  | Entry x coordinate (cartesian projected position)
| y_entry       | Double  | Entry y coordinate (cartesian projected position)
| x_exit        | Double  | Exit x coordinate (cartesian projected position)
| y_exit        | Double  | Exit y coordinate (cartesian projected position)

Vmax Vmin and Vmean may not be recorded.

The task is to predict the location of this last exit point and whether this device is within the city center or not. The target variable is the latter. 

## Some terminologies 
We use the following terms in bold to make our explanation more intuitive
- The **last seen location** are is the coordinates of x_entry and y_entry of the final trajectory. While different devices have a different number of trajectory, all devices have a final trajectory and hence a last seen location. 
- If the "last seen location" of the device is outside the boundary, we say that the device is **last seen inside**. Otherwise, the device is **last seen outside**. 
- If the last seen duration (time_entry - time_exit) is zero, we say the device is **stationary**. Otherwise, the device is **not stationary**. Note that in the use of this term we are concerned with the last trajectory.
- The **final location** is the co-ordinates of the last exit point. Note that this is not given in the test set.
- If the final location is inside the boundary, we say that the device **ends up inside**. Otherwise, the device **ends up outside**. We should return a target of 1.0 in our submission if we predict the device to end up inside; otherwise, we should return a value of 0.0.

## Basic analysis
- If we predict all the points to end up inside, you will get a public F1 score of 0.43379. This is indicative of the proportion of points that end up inside in the test set.

The following shows the distributions on whether the points are last seen inside or outside, stationary or not stationary.

##### Training set

| Quantity       | Last seen inside | Last seen outside | Total
| ---------------|------------------|-------------------|------
| Stationary     | 19512            | 47486             | **66998**
| Not stationary |                  |                   | **67065**
| **Total**      | **38449**        | **95614**         | 

##### Testing set

| Quantity       | Last seen inside | Last seen outside | Total
| ---------------|------------------|-------------------|------
| Stationary     |                  |                   | **16841**
| Not stationary |                  |                   | **16674**
| **Total**      | **9544**         | **23971**         | 

#### Naive predictions
If the point is stationary, the last seen location is the final location. 
If we predict the final location to be the last seen location, we get an F1 score of **0.86881**. 
Surprisingly, at the end, many participants did not achieve such a score.

## Visualisations
To add.

## Kernels
Preprocessing and training is done on Kaggle, which provides free storage and computing services. 
The python code is run on Jupyter notebooks, which are called "kernels" on Kaggle. 
Kaggle kernels also have a version control system that allows different versions of the notebook and results to be saved.
For documentation, we log the public leaderboard results in the comment.

### Overview of kernels
To add.

### List of kernels
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

#### Supplementary kernels for visualisation
Separate models for stationary points that are last seen inside and last seen outside
https://www.kaggle.com/huikang/eyn-df-lgbm-citysplit

To visualise the distribution of starting and ending points
https://www.kaggle.com/huikang/eyn-df-map

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
- Parameter search. Requires script writing and resources more extensive than Kaggle.
- Learn the theory behind the F1 score and the optimal threshold.
- `eyn-original` takes 5 hours to run due to creation of numerous dataframes.

### Learning points
- Use of dataframes is worth the time investment, rather than working on numpy arrays.
- The problem of overfitting.
  - The best public leaderboard score may not be the best private leaderboard score. 
  - Trusting your own CV score, avoid training on your validation set 
