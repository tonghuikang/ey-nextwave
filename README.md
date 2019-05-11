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
- If the **final duration** (time_exit - time_entry) is zero, we say the device is **stationary**. Otherwise, the device is **not stationary**. Note that in the use of this term we are concerned with the last trajectory.
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
The stationary points has its last seen location equal to its final location.
![map_stationary.png](./map_stationary.png)

The following two plots are for non-stationary points.
- This following plot of non-stationary points emphasises those with a shorter final duration under 1 hour
![map_nonstat_shortdur.png](./map_nonstat_shortdur.png)

- This following plot of non-stationary points emphasises those with a longer final duration
![map_nonstat_longdur.png](./map_nonstat_longdur.png)

We can see that the roads resembling veins are more visible in the scatter plot of non-stationary points. 
The above visualisations in produced in https://www.kaggle.com/huikang/eyn-df-map

## Kernels
Preprocessing and training is done on Kaggle, which provides free storage and computing services. 
The python code is run on Jupyter notebooks, which are called "kernels" on Kaggle. 
Kaggle kernels also have a version control system that allows different versions of the notebook and results to be saved.
For documentation, we log the public leaderboard results in the comment section.

### Overview of kernels
We have separate kernels for separate functions so that preprocessing need not be repeated. 
It also makes the pipeline more stable upstream data are frozen with version control.

The following chart shows the order of how the kernels are run. 
Each kernel depends on one or more kernels from the above.
![kernels_overview.png](./kernels_overview.png)

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

To visualise the distribution of the last seen location and whether it ends up inside
https://www.kaggle.com/huikang/eyn-df-map

## Solution and Discussion
We discuss our approach and reflections.

### Features of our solution
- **Transformation into an array of 21\*13 elements** for each "device" from the source dataset. The standardisation allows for easier data processing downstream.
- **Reversed order of trajectory** in the array. With the last trajectory is always at position zero, the last seen location, final duration and always at the same location. 
- **10-fold cross validation**. This is a standard for datasets with this amount of data. The folds, which are stratified, are defined in `eyn_folds`.
- **Use of the best threshold from the training set**. We find the best threshold from the cross-validated training data instead of 0.5. More theory or evidence is necessary to understand the merit of this measure.
- **Retaining NaNs in dataframes.** We do not fill the NaN with numbers like zero, which has a different meant. It helps as LGBM allows NaNs as input.
- **Training features**
  - **Trival features** We computed the duration, speed and distance for each trajectory. We also computed the duration, speed and distance between adjacent trajectory (this entry point and the previous trajectory's exit point).
  - **Cluster features** We place the last seen location of the non-stationary points on a map. (As we suspect that the projected coordinates are warped, the y-coordinates is scaled down by a factor of 10). To generate the cluster features for each device analysed, we use its last seen location to find the nearest 10 last seen locations. Naturally, the first last seen location is itself, so we remove this result. This above is done in `eyn_df_cluster_unif`. 
  Then we calculate a weighted average of the result. Points further away from the analysed last seen location have exponentially lower weights. This is done in `eyn_df_cluster_calc`.
  - **Embeddings as a feature** Neural networks could not produce results better than the baseline of predicting final location to be its initial location. However, we believe some learning can be transferred from a neural network. LGBM only allows for training to one target, but neural networks can be trained on multiple classes at once. The neural network model thus also trains on the final location as well as the augmented trivial features. The final layer of the neural network presents embeddings of size 15, and this is introduced to the LGBM model.
- **Use of LightGBM** There parameters are crudely tuned. Notably, the best `num_leaves` is under 100, and we used 63 eventually.

### Benefits of our solution
- No additional information used
- Consistent documentation throughout (with the shape of array printed)
- Fast training time and preprocessing time (except processing dataset into a numpy array and clustering last seen locations)
- Visualisation of more important features
- Competitive public score of 0.89441

### Analysis of the importance of features
To be added

### What could have done better
- More systematic documentation. Understand how much each feature contribute how much to the score.
- Parameter search. Requires script writing and resources more extensive than Kaggle.
- Learn the theory behind the F1 score and the optimal threshold.
- `eyn-original` takes 5 hours to run due to the creation of numerous dataframes.

### Learning points
- Use of dataframes is worth the time investment, rather than working on numpy arrays.
- The problem of overfitting.
  - The best public leaderboard score may not be the best private leaderboard score. 
  - Trusting your own CV score, avoid training on your validation set 
