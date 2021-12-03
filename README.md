# Machine-Learning-Algorithms

# 1. Implementing Multiple Linear Regression
## Objective:
To predict the profit made by a startup on the basis of expenses incurred and the state where they operate

accuracy = 93.47%
# 2. Bias_Variance_Tradeoff:
# 3. Project_Linear_Regression (CAR_price)
## Problem Statement:
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.

They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:

Which variables are significant in predicting the price of a car How well those variables describe the price of a car Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.

## Business Goal:
We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

# 4.Cross Validation & Hyperparameter Tuning:

## Problem Statement
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
Which variables are significant in predicting the price of a car
How well those variables describe the price of a car
Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market.

## Business Goal
We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

### The Cross-Validation Procedure
![image](https://user-images.githubusercontent.com/77626222/142729298-273a51c7-8ae9-4557-8fde-3f275cb6d04a.png)
The Cross-Validation Procedure
In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality. For example, we could have 5 folds or experiments. We divide the data into 5 pieces, each being 20% of the full dataset.
We run an experiment called experiment 1 which uses the first fold as a holdout set, and everything else as training data. This gives us a measure of model quality based on a 20% holdout set, much as we got from using the simple train-test split.
We then run a second experiment, where we hold out data from the second fold (using everything except the 2nd fold for training the model.) This gives us a second estimate of model quality. We repeat this process, using every fold once as the holdout. Putting this together, 100% of the data is used as a holdout at some point.
Returning to our example above from train-test split, if we have 5000 rows of data, we end up with a measure of model quality based on 5000 rows of holdout (even if we don't use all 5000 rows simultaneously.

# 5. Logistic Regression

![image](https://user-images.githubusercontent.com/77626222/143466927-e6c988c5-93d4-4e0b-95b0-5a803fb4c2f4.png)

Logistic regression is a classification algorithm that predicts the probability of an outcome that can only have two values (i.e. a dichotomy). A logistic regression produces a logistic curve, which is limited to values between 0 and 1. Logistic regression models the probability that each input belongs to a particular category.
Logistic regression is an excellent tool to know for classification problems, which are problems where the output value that we wish to predict only takes on only a small number of discrete values. Here we'll focus on the binary classification problem, where the output can take on only two distinct classes.

# 6. Decision Tree

Decision tree is a type of supervised learning algorithm that is mostly used in classification problems. It works for both categorical and continuous input and output variables.

Example:-

Let’s say we have a sample of 30 students with three variables Gender (Boy/Girl), Class(IX/X) and Height (5 to 6 ft). 15 out of these 30 play cricket in leisure time. Now, I want to create a model to predict who will play cricket during leisure period? In this problem, we need to segregate students who play cricket in their leisure time based on highly significant input variable among all three.

This is where decision tree helps, it will segregate the students based on all values of three variable and identify the variable, which creates the best homogeneous sets of students (which are heterogeneous to each other). In the snapshot below, you can see that variable Gender is able to identify best homogeneous sets compared to the other two variables.
![image](https://user-images.githubusercontent.com/77626222/143275864-ac42ef85-38dd-49a2-ae23-47c2d54ca3b6.png)
## Important Terminology
Root Node: It represents entire population or sample and this further gets divided into two sets.

Splitting: It is a process of dividing a node into two sub-nodes.

Decision Node: When a sub-node splits into further sub-nodes, then it is called decision node.

Leaf/ Terminal Node: Nodes do not split is called leaf or terminal node.

Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.

Branch / Sub-Tree: A sub section of entire tree is called branch or sub-tree.

Parent and Child Node: A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.

![image](https://user-images.githubusercontent.com/77626222/143276233-c18d888f-8b87-4a78-aa86-f1a4c3315940.png)

## Advantages
Easy to Understand: Decision tree output is very easy to understand even for people from non-analytical background. It does not require any statistical knowledge to read and interpret them. Its graphical representation is very intuitive and users can easily relate their hypothesis.

Useful in Data exploration: Decision tree is one of the fastest way to identify most significant variables and relation between two or more variables. With the help of decision trees, we can create new variables / features that has better power to predict target variable. It can also be used in data exploration stage. For example, we are working on a problem where we have information available in hundreds of variables, there decision tree will help to identify most significant variable. Less data cleaning required: It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree.

Non Parametric Method: Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.

## Disadvantages
Over fitting: Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning.

# 7. Ensembles of Decision Trees:

The literary meaning of word ‘ensemble’ is group. Ensemble methods involve group of predictive models to achieve a better accuracy and model stability.

Ensemble methods are known to impart supreme boost to tree based models.

Like every other model, a tree based algorithm also suffers from the plague of bias and variance.

Decision trees are prone to overfitting.

Normally, as you increase the complexity of your model, in this case decision tree, you will see a reduction in training error due to lower bias in the model. As you continue to make your model more complex, you end up over-fitting your model and your model will start suffering from high variance.

A champion model should maintain a balance between these two types of errors. This is known as the trade-off management of bias-variance errors.

Ensemble learning is one way to tackle bias-variance trade-off.

There are various ways to ensemble weak learners to come up with strong learners:

Bagging
Boosting
Stacking

## 1. Bagging

![image](https://user-images.githubusercontent.com/77626222/143467234-5cf1d6d6-a792-4f37-b765-ca15f084823f.png)

Bagging is an ensemble technique used to reduce the variance of our predictions by combining the result of multiple classifiers modeled on different sub-samples of the same data set.

### Random Forest
In Random Forest, we grow multiple trees as opposed to a single tree in CART model.
We construct trees from the subsets of the original dataset. These subsets can have a fraction of the columns as well as rows.
To classify a new object based on attributes, each tree gives a classification and we say that the tree “votes” for that class.
The forest chooses the classification having the most votes (over all the trees in the forest) and in case of regression, it takes the average of outputs by different trees.

![image](https://user-images.githubusercontent.com/77626222/143467426-6bc43079-821b-4403-a354-2d8bb0cb2014.png)

### Advantages
This algorithm can solve both type of problems i.e. classification and regression and does a decent estimation at both fronts.
RF has the power of handling large datasets with higher dimensionality. It can handle thousands of input variables and identify most significant variables so it is considered as one of the dimensionality reduction methods. Further, the model outputs Importance of variable, which can be a very handy feature (on some random data set).
It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data is missing.
It has methods for balancing errors in data sets where classes are imbalanced.
The capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
Random Forest involves sampling of the input data with replacement called as bootstrap sampling. Here one third (say) of the data is not used for training and can be used to testing. These are called the out of bag samples. Error estimated on these out of bag samples is known as out of bag error. Out-of-bag estimate is as accurate as using a test set of the same size as the training set. Therefore, using the out-of-bag error estimate removes the need for a set aside test set.

### Disadvantages
It surely does a good job at classification but not as good as for regression problem as it does not give precise continuous nature predictions. In case of regression, it doesn’t predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy.
Random Forest can feel like a black box approach for statistical modelers – you have very little control on what the model does. You can at best – try different parameters and random seeds!

## 2. Boosting

![image](https://user-images.githubusercontent.com/77626222/143467631-7d5c81bd-f8f0-4454-bb0c-a15c0b1218e9.png)

Boosting fit a sequence of weak learners − models that are only slightly better than random guessing, such as small decision trees − to weighted versions of the data. More weight is given to examples that were misclassified by earlier rounds.

There are many boosting algorithms which impart additional boost to model’s accuracy:

Gradient Boosting Machine
XGBoost
AdaBoost
LightGBM
CatBoost

## 3. Stacking

Stacking or Stacked Generalization is an ensemble technique.

It uses a meta-learning algorithm to learn how to best combine the predictions from two or more base machine learning algorithms.

The benefit of stacking is that it can harness the capabilities of a range of well-performing models on a classification or regression task and make predictions that have better performance than any single model in the ensemble.

Given multiple machine learning models that are skillful on a problem, but in different ways, how do you choose which model to use (trust)?

The approach to this question is to use another machine learning model that learns when to use or trust each model in the ensemble.

Unlike bagging, in stacking, the models are typically different (e.g. not all decision trees) and fit on the same dataset (e.g. instead of samples of the training dataset).

Unlike boosting, in stacking, a single model is used to learn how to best combine the predictions from the contributing models (e.g. instead of a sequence of models that correct the predictions of prior models). The architecture of a stacking model involves two or more base models, often referred to as level-0 models, and a meta-model that combines the predictions of the base models, referred to as a level-1 model.

Level-0 Models (Base-Models): Models fit on the training data and whose predictions are compiled.

Level-1 Model (Meta-Model): Model that learns how to best combine the predictions of the base models. The meta-model is trained on the predictions made by base models on out-of-sample data. That is, data not used to train the base models is fed to the base models, predictions are made, and these predictions, along with the expected outputs, provide the input and output pairs of the training dataset used to fit the meta-model.

The outputs from the base models used as input to the meta-model may be real value in the case of regression, and probability values, probability like values, or class labels in the case of classification.

## Bagging vs. Boosting

![image](https://user-images.githubusercontent.com/77626222/143467856-5fa5aeb6-7c8f-49d6-87ff-c5db1f18df89.png)

# 8. KNN:
K-nearest neighbors (kNN) is a supervised machine learning algorithm that can be used to solve both classification and regression tasks.

![image](https://user-images.githubusercontent.com/77626222/144539450-fcd7bd3e-52da-49f3-a2b0-ba540b274bf1.png)

### How does kNN work?
The kNN working can be explained on the basis of the below algorithm:

    Step-1: Select the number K of the neighbors
    Step-2: Calculate the Euclidean distance of K number of neighbors
    Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
    Step-4: Among these k neighbors, count the number of the data points in each category.
    Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
    Step-6: Our model is ready.
    
### Advantages of KNN Algorithm:
    It is simple to implement.
    It is robust to the noisy training data
    It can be more effective if the training data is large.
### Disadvantages of KNN Algorithm:
    Always needs to determine the value of K which may be complex some time.
    The computation cost is high because of calculating the distance between the data points for all the training samples.
