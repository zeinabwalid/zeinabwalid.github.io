---
title: "R Machine Learning Project-Cancer Classification "
layout: post
date: 2019-12-02 22:44
image: /assets/images/markdown.jpg
headerImage: false
tag:
- Tumor
- TCGA
- Machine Learning
- R programming Language
star: true
category: blog
author: johndoe
description: Markdown summary with different options
---
***By:[Mohammed Elsayed](https://mohammedelsayed412.github.io)--[Galal Hossam](https://galal-hossam-eldien.github.io/Galal-Hosam-Eldien/)--[Zeinab Walid](http://zeinabwalid.github.io)--[Esraa Sayed](https://esraasayed98.github.io/EsraaSayed.github.io/)***
<br>
***Supervised by:Dr.Ayman EL-Dieb --Eng.Asem Alaa*** 
# Abstract

&nbsp;&nbsp;&nbsp;&nbsp;One day, we decided to visit an oncology hospital , we saw different patients 
with different types of tumors and serious cases , we couldn’t imagine what we saw,
hearing is totally different from seeing people suffering.<br>
&nbsp;&nbsp;&nbsp;&nbsp;There is no doubt that malignant tumor is a very dangerous disease and nowadays
it spreads dramatically all over the world specially in Egypt , this is an alarm to
condense our efforts to study such a dangerous disease to enhance therapies for 
different types of tumor.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Our project is to analyze different numbers of humans tumors to discover 
molecular aberrations of DNA ,RNA , protein and epigenetic levels , this data
provide opportunity to develop integrated picture of commonalities ,differences 
and emergent themes across tumor lineages, we compare five types of tumor, 
Colon (COAD) – Breast (BRCA) – Lung adenocarcinoma (LUAD) – Kidney (KIRC)-Prostate
adenocarcinoma (PRAD),  profiled by "The Cancer Genome Atlas Pan-Cancer analysis"
(TCGA),this analysis of molecular aberrations and their functional roles across 
tumor types will teach us how to extend therapies effective in one tumor type to
others with a similar genomic profile.<br>
&nbsp;&nbsp;&nbsp;&nbsp;We are looking forward to getting the most benefit from this study using 
different algorithms using R  programming language  to know the best algorithm that fits our dataset
 with the best accuracy to contribute with a little thing to decrease the danger of this disease.<br>




# Data 
   
Study the paper explains the dataset of the project which is <br>
[The Cancer Genome Atlas Pan-Cancer analysis project](https://www.nature.com/articles/ng.2764)

# Data Preprocessing : <br>
   We face lots of challenges in data preprocessing mainly in feature selection, as we work on TCGA
   dataset and it is considered a huge dataset as it has more than 16000 features and about 
   800 samples.<br>
### a) Feature Normalization:<br> 
The value of our dataset are near from each other , so we don’t need to 
make feature scaling ,as features normalization is important to make data near to each other to
get the best performance.<br>
### b) Data imputation: <br>
 we have no missing data.<br>
### c) Feature Selection: <br>
We worked on feature selection to select the most significant features from 
this massive dataset.<br>
There are lots of statistical techniques can be used to reduce the redundant data.<br>
Using only the non-redundant data improves the model performance, makes the training of the
model faster, and reduces the complexity of the model.<br>
We tried to make feature selection using two methods which are:<br>
#### 1. Boruta Algorithm:<br>
It is a feature selection method, it works as a wrapper algorithm built around Random Forest
Classification Algorithm, it tries to capture all the important features in your dataset.<br>
***How does it work?*** <br>
It duplicates the dataset by creating shuffled copies of all features which are called shadow 
features. <br>
Then it trains a random forest classifier on the extended dataset and select features according
to its importance (the default is mean decrease accuracy) to evaluate the importance of each
feature where higher means more important.<br>
At every iteration, the algorithm checks whether a real feature has a higher z-score than the 
maximum z-score of its shadow features, if they do, it records this in a vector which are called
hits, next it will contribute with another iteration.<br>
Finally, the algorithm stops either when all feature gets confirmed or rejected or it reaches
a specified limit of random forest runs.<br>
Here is some parts of our code <br>
* Used libraries:<br>
		 
         library(Boruta)
         library(mlbench)
		 library(caret)
         library(randomForest)
           
* Uploading dataset :	<br>			
			
         dataset = read.csv('TCGA_data.csv')
		 
* Labeling:<br>

		 dataset$Class = factor(dataset$Class,levels = c('PRAD' , 'LUAD' ,'BRCA','KIRC','COAD'),
		 labels = c(1,2,3,4,5))
   	
* Applying Boruta algorithm :<br>
        
         set.seed(111)
		 boruta <- Boruta(Class ~ ., data = data, doTrace = 2, maxRuns = 1000)
         print(boruta)
         
             
We didn’t continue our project with this method because its run time was very big with our
dataset as it took more than 2 days and it needs a lot of memory,So we used a better method 
which is the principal component analysis.<br>
#### 2. Principal Component Analysis (PCA):<br>
 it is a useful technique for exploratory data analysis, it’s helpful with a large dataset 
 which have many variables.<br>
PCA is a dimensionality reduction or a data compression method that can be used to reduce
a large set of variables to a small set ”principal components ”  that still contains most of 
the information in the large set.<br>
***How does it work?***<br>
you take a dataset with many variables, and you simplify that dataset by turning your original 
variables into a smaller number of "Principal Components".<br>
Principal Components are the underlying structure in the data. They are the directions where
there is the most variance, the directions where the data is most spread out.<br>
PCA is a type of linear transformation on a given data set that has values for a certain number
of variables (coordinates) for a certain amount of spaces. This linear transformation fits this
dataset to a new coordinate system in such a way that the most significant variance is found on
the first coordinate, and each subsequent coordinate is orthogonal to the last and has a lesser
variance. In this way, you transform a set of x correlated variables over y samples to a set of
p uncorrelated principal components over the same samples.<br>
for more information about PCA ,kindly open this:
[PCA](https://www.datacamp.com/community/tutorials/pca-analysis-r)<br><br>
	
* Importing data set :
 
         data1=read.csv('TCGA_data.csv')
  
* There were a columns which have zero values and this method couldn’t compute the variance for
	it,so we eliminated them:
			    
			data=data1[,-c(7,25,4372,4810,4811,4373,4377,4379,4830,4816,4817,4818,4819,4821,4825,
		    4833,5290,7663,7664,7665,7666,7667,8123,9306,9308,9316,9318,9322,9325,
		    9454,10123,11960,12647,13993,14160,14161,14163,15140,15142,15143,15448,12720,13862)]

			 
			
* Encoding Catergorical data:
			 			
			 data$Class = factor(data$Class,
			 levels = c('PRAD' , 'LUAD' ,'BRCA','KIRC','COAD'),
			 labels = c(1,2,3,4,5))
				
* Applying PCA method:
			library(caret)
			library(e1071)

			 pca= preProcess(x=data[-1] , method='pca',pcaComp = n)  
			 new_data=predict(pca,data)

n refers to the of principal componats to get the highest accuracy , but it is different for each algorithm ,we will dicuss how get it:

As an example we will dicuss naive bayes algorithm for getting n :

So The most fitted number of PCs for naive bayes on our data was 27 with accuracy 0.98125, 
and we apply the smae steps for other methods and get 20 for decision tree , and 30 for K-nn.<br>

# Model Validation <br>
Used cross-validation methods for assessing model performance:<br>

1- **The Validation set Approach(data split)**:<br>
The validation set approach consists of randomly splitting the data into two sets: <br>
one set is used to train the model and the remaining other set is used to test the model.<br>
The process works as follow:<br>
  1. Train the model on the training data set<br>
  2. Apply the model to the test data set to predict the outcome of new unseen observations<br>

* We have used 80%of our data for training the model  and 20% is used to evaluate the model performance(test data ).

		library(caTools)
		set.seed(123)
		split = sample.split(new_data$Class , SplitRatio=0.8)
		training_set = subset(new_data , split == TRUE)
		test_set = subset(new_data , split == FALSE)

2- **K-fold cross-validation**<br>
The k-fold cross-validation method evaluates the model performance on different subset of the training data <br> 
and then calculate the average of the accurses of the different test data.<br>

* Applying cross validation on algorithm as follow:<br>
1-	Randomly split the data into k-folds.<br>
2-	For each k-fold in your dataset, build your model on k – 1 folds of the dataset. <br> 
    Then, test the model to check the effectiveness for kth fold.<br>
3-	Repeat this process until each of the k subsets has served as the test set.<br>
4-	Check the accuracy of the model by computing the average accuracy of each iteration.<br>

For more information about Model Validation, you could see [Model validation](http://www.sthda.com/english/articles/38-regression-model-validation/157-cross-validation-essentials-in-r/).



# Methodology:  <br>
### a) Naive Bayes :  <br>
It is a classification technique based on Bayes’ Theorem with an assumption of independence
 among predictors.<br> In simple terms, a Naive Bayes classifier assumes that the presence of
 a particular feature in a class is unrelated to the presence of any other feature, it deals 
 with conditional probability so it can study the relation between features and how to deal
 with it to classify our data and can make the prediction.<br>
Naive Bayes model is easy to build and particularly useful for very large data sets.<br> 
Along with simplicity, Naive Bayes is known to outperform even highly sophisticated classification methods.<br>
So, we used it in our project to deal with our large data, it helps us to deal with five classifications of cancer with high accuracy.
* Output data of PCA with n = 27 :<br>

		library(caret)
		library(e1071)
		pca= preProcess(x=data[-1] , method='pca',pcaComp = 27)  
		new_data=predict(pca,data)
* Applying naïve Bayes after splitting: <br>

		 library(e1071)
		 classifier =  naiveBayes(x = training_set[-1],
                         y = training_set[[1]])
		 y_pred = predict(classifier, newdata = test_set[-1])

		
* Confusion matrix and Accuracy:<br> 

		cm = table(test_set[[1]], y_pred)
		accuracy = ((cm[1,1] + cm[2,2] + cm[3,3]) + cm[4,4] + cm[5,5]) / (sum(cm))	
		accuracy

* Applying cross validation:<br> 

				library(caret)
				folds=createFolds(new_data$Class,k=10)
				cv = lapply(folds, function(x) {
				training_fold = new_data[-x, ]
				test_fold = new_data[x, ]
				classifier =  naiveBayes(x = training_fold[-1],
								   y = training_fold[[1]])

				y_pred = predict(classifier, newdata = test_fold[-1])
				cm = table(test_fold[, 1], y_pred)
				accuracy = (cm[1,1] + cm[2,2] + cm[3,3] + cm[4,4] + cm[5,5]) / 
				(sum(cm))
				return(accuracy)
				<br>
				<br>
* Confusion matrix and TOT_Accuracy:<br>

		       accuracy_tot = mean(as.numeric(cv))
		       accuracy_tot
		
### b) Decision Tree <br>
Decision tree is the most powerful and popular tool for classification and prediction.<br>
Decision tree is a flowchart like tree structure, where each internal node denotes a test on an attribute,4
each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label.<br>
IT classifies instances by sorting them down the tree from the root to some leaf node, which provides the 
classification of the instance. <br>
An instance is classified by starting at the root node of the tree testing the attribute specified by <br>
this node then moving down the tree branch corresponding to the value of the attribute Decision trees <br> 
are able to generate understandable rules,  perform classification without requiring much computation,<br>
are able to handle both continuous and categorical variables, provide a clear indication of which fields<br>
are most important for prediction or classification.<br>

* Output data of PCA with n = 20 :<br>

		 library(caret)
		 library(e1071)
		 pca= preProcess(x=data[-1] , method='pca',pcaComp = 20)  
		 new_data=predict(pca,data)
* Applying Decision Tree after splitting: <br>

		 library(e1071)
		 classifier=rpart(formula = Class ~ . , data=training_set)
		 y_pred = predict(classifier, newdata = test_set[,-1],type='class')
		
* Confusion matrix and Accuracy:<br>
 
		cm = table(test_set[[1]], y_pred)
		accuracy = ((cm[1,1] + cm[2,2] + cm[3,3]) + cm[4,4] + cm[5,5]) / (sum(cm))	
		accuracy
* Applying cross validation:<br> 

		library(caret)
		folds=createFolds(new_data$Class,k=10)
		cv=lapply(folds,function(x_fold){
		  training_fold=new_data[-x_fold,]
		  test_fold=new_data[x_fold,]
		  classifier=rpart(formula=Class~. , data=training_fold)
		  y_pred = predict(classifier, newdata = test_fold[-1],type='class')
		  cm = table(test_fold[,1], y_pred)
		  accuracy=((cm[1,1]+cm[2,2]+cm[3,3]+cm[4,4]+cm[5,5])/ sum(cm))
		  return(accuracy)
		})
* Confusion matrix and TOT_Accuracy :<br>

		tot_acc=mean(as.numeric(cv))
		tot_acc
		
###  c)" k-nearest-neighbour" KNN <br>
A k-nearest-neighbour is a data classification algorithm that attempts to determine what group a data point<br>
is in by looking at the data points around it.<br>
An algorithm, looking at one point on a grid, trying to determine if a point is in group A or B, looks at the<br>
states of the points that are near it.<br>
The range is arbitrarily determined, but the point is to take a sample of the data.<br>
If the majority of the points are in group A, then it is likely that the data point in question will
be A rather than B, and vice versa.<br>
The k-nearest-neighbour is an example of a "lazy learner" algorithm because it does not generate a model of the <br>
data set beforehand. <br>
The only calculations it makes are when it is asked to poll the data point's neighbours.<br> This makes k-nn very easy
to implement for data mining.<br>

* Output data of PCA with n = 30 :<br>

		 library(caret)
		 library(e1071)
		 pca= preProcess(x=data[-1] , method='pca',pcaComp = 30)  
		 new_data=predict(pca,data)
* Applying Decision Tree after splitting: <br>

		 library(class)
		y_pred = knn(train = training_set[, -1],
				 test = test_set[, -1],
				 cl =training_set$Class,
				 k = 5,
				 prob = TRUE)		
* Confusion matrix and Accuracy:<br> 

		cm = table(test_set[[1]], y_pred)
		accuracy = ((cm[1,1] + cm[2,2] + cm[3,3]) + cm[4,4] + cm[5,5]) / (sum(cm))	
		accuracy
* Applying cross validation:<br> 

		library(lattice)
		library(caret)
		folds=createFolds(new_data$Class,k=10)
		cv = lapply(folds, function(x) {
		  training_fold = new_data[-x, ]
		  test_fold = new_data[x, ]
		  y_pred = knn(train = training_fold[, -1],
					   test = test_fold[, -1],
					   cl = training_fold[[1]],
					   k = 5,
					   prob = TRUE)
		  cm = table(test_fold[[ 1]], y_pred)
		  accuracy = (cm[1,1] + cm[2,2] + cm[3,3] + cm[4,4] + cm[5,5]) / 
			(sum(cm))
		  return(accuracy)
		})
* Confusion matrix and TOT_Accuracy :<br>

		tot_acc=mean(as.numeric(cv))
		tot_acc

# Result

### Naive Bayes <br>
***PCA iterations***

<image src = "/assets/images/tableaccnaive.PNG"/>
SO we chossed number of PCA componat = 27 for naive bayes.

#### Using Validation set Approach  <br>

* confusion matrix	

<image src = "/assets/images/cmnaive.PNG"/>

* accuracy <br>


<image src = "/assets/images/accnaive.PNG"/>

***Using cross validation***  <br>

* confusion Matrix <br> 	

<p><image src = "/assets/images/cmcrossnaive.PNG"/></p>
* total accuracy <br>
		
<image src = "/assets/images/totalacc.PNG"/><br>


### Decision Tree <br>

***Using Validation set Approach***  <br>

* confusion matrix  <br>

<image src = "/assets/images/cmdec.PNG"/><br><br>


* accuracy <br>


<image src = "/assets/images/accdec.PNG"/><br>


***Using cross validation*** <br>

* confusion Matrix <br> 	

<image src = "/assets/images/cmcrossdec.PNG"/>

* total accuracy <br>
		
<image src = "/assets/images/totalaccdec.PNG"/>


### KNN <br>

***Using Validation set Approach*** 

***confusion matrix*** 

<image src = "/assets/images/kcm.PNG"/>

* accuracy <br>


<image src = "/assets/images/kacc.PNG"/>


***Using cross validation***  

* confusion Matrix 	

<image src = "/assets/images/kcmcross.PNG"/>


* total accuracy 
		
<image src = "/assets/images/ktotalacc.PNG"/>



# Learning Outcomes

* Dealing with genes ratios and how to use it in cancer detection.
* Understanding what the machine learning is and its different types.
* Understanding four types of machine learning algorithms of classification and regression.
* How to choose the best algorithm fitting to any data.
* Machine learning concepts like: over fitting, cross validation, data splitting, factorization…etc.
* Data pre-processing different concepts and different data types, i.e. categorical and numerical. 
* Dealing with big data features and how to regulate this data.


# Problems

In our project we faced some problems ,we tried to fix some of them as we discussed before in feature selectin.<br>
Here are two other problems:<br>
* Visualization:<br>
We try to visualize our data with many ways but we have problems with the size of the data and we have five<br>
classes to predict, so we can't visualize our data in 2D plane.<br>
* Logistic regression: <br>
We are trying to apply this algorithm on our data, but  we found that logistic regression only applied on two<br>
output classes and we have five classes so we can't apply this algorithm to our data.<br>    
 

 

# Future Work 

* Studying how to convert genes sequence to ratios like our data to know more details about  the reference of data.<br>
* Applying other algorithms to our data and study their performance until reach to the best model.
* Visualize our data by another powerful machine that can adapt to our large data.
* Applying logistics regression to more than two output classes.  
* Try to perform mobile application that can help doctors to deal with the output results easily.  




		
# Conclusion
&nbsp;&nbsp;&nbsp;&nbsp;***"Don't fear failure ,but rather fear not trying "*** This is our believe 
during our work on this project,although it is our first project in machine learning,we challenge
ourselves to deal with this enormous dataseand we keep trying to get the best of us , we used 
different algorithms to compare between them on the dataset which is essential in discovering 
different kinds of therapies to five types of tumor. 



