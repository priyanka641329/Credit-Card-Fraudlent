
# Credit Card Fraudlent




## Context
   It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
## Content
   The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

   It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
## Inspiration
  Identify fraudulent credit card transactions.

  Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
## Acknowledgement

 - https://www.researchgate.net/project/Fraud-detection-5 



## Dataset
  https://www.kaggle.com/mlg-ulb/creditcardfraud
  
## Model Prediction
Now it is time to start building the model .The types of algorithms we are going to use to try to do anomaly detection on this dataset are as follows...
## Isolation Forest Algorithm :

One of the newest techniques to detect anomalies is called Isolation Forests. The algorithm is based on the fact that anomalies are data points that are few and different. As a result of these properties, anomalies are susceptible to a mechanism called isolation.

This method is highly useful and is fundamentally different from all existing methods. It introduces the use of isolation as a more effective and efficient means to detect anomalies than the commonly used basic distance and density measures. Moreover, this method is an algorithm with a low linear time complexity and a small memory requirement. It builds a good performing model with a small number of trees using small sub-samples of fixed size, regardless of the size of a data set.

Typical machine learning methods tend to work better when the patterns they try to learn are balanced, meaning the same amount of good and bad behaviors are present in the dataset.

How Isolation Forests Work The Isolation Forest algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic argument goes: isolating anomaly observations is easier because only a few conditions are needed to separate those cases from the normal observations. On the other hand, isolating normal observations require more conditions. Therefore, an anomaly score can be calculated as the number of conditions required to separate a given observation.

The way that the algorithm constructs the separation is by first creating isolation trees, or random decision trees. Then, the score is calculated as the path length to isolate the observation.
## Local Outlier Factor(LOF) Algorithm:

The LOF algorithm is an unsupervised outlier detection method which computes the local density deviation of a given data point with respect to its neighbors. It considers as outlier samples that have a substantially lower density than their neighbors.

The number of neighbors considered, (parameter n_neighbors) is typically chosen 1) greater than the minimum number of objects a cluster has to contain, so that other objects can be local outliers relative to this cluster, and 2) smaller than the maximum number of close by objects that can potentially be local outliers. In practice, such informations are generally not available, and taking n_neighbors=20 appears to work well in general.

## Observations :

1) Isolation Forest detected 73 errors versus Local Outlier Factor detecting 97 errors vs. SVM detecting 8516 errors

2) Isolation Forest has a 99.74% more accurate than LOF of 99.65% and SVM of 70.09

3) When comparing error precision & recall for 3 models , the Isolation Forest performed much better than the LOF as we can see that the detection of fraud cases is around 27 % versus LOF detection rate of just 2 % and SVM of 0%.

4) So overall Isolation Forest Method performed much better in determining the fraud cases which is around 30%.

5) We can also improve on this accuracy by increasing the sample size or use deep learning algorithms however at the cost of computational expense.We can also use complex anomaly detection models to get better accuracy in determining more fraudulent cases.
## Screenshots

![Representation of Data](https://github.com/priyanka641329/Credit-Card-Fraudlent/assets/84657912/4478a0e4-1ffd-43cc-bae5-883ce90f1603)



![Data](https://github.com/priyanka641329/Credit-Card-Fraudlent/assets/84657912/12b8f839-e3ee-419b-8848-aa853c0c8e5e)



![Data 1](https://github.com/priyanka641329/Credit-Card-Fraudlent/assets/84657912/3250fcf5-1bb0-43f8-8910-a7f954d7ecc6)



![Correlation](https://github.com/priyanka641329/Credit-Card-Fraudlent/assets/84657912/7dcf679d-e792-45f1-901b-f0dcf2e2e996)
## Running Tests



![Running test](https://github.com/priyanka641329/Credit-Card-Fraudlent/assets/84657912/39b48afc-f55e-4027-8d8f-774bc186fccf)





![SVM](https://github.com/priyanka641329/Credit-Card-Fraudlent/assets/84657912/2f095e93-cdb7-4190-b478-2a843e49d14a)

