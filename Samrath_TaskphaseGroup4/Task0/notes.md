# Concepts related to Machine learning

1.       
Data: Data is essential for the working of an ML
model. Related data is stored in datasets. Datasets contain 2 parts, a label
and the features. Features ate the values given to the model and the predicted
value is the label. Supervised learning receives labelled datasets while
unsupervised learning receives unlabeled data sets to which it assigns the
labels.

Characteristics
of a dataset- A dataset has 2 major characteristics, being the size and
diversity. An Ideal dataset has a large size and is also very diverse.

Datsets
may also have various number of features. More features allow for more indepth
and better predicting models, but too many features may cause no causation
matches and hence lead to false results.

2.      
Model: The model is a complex collection of values
which maps the relationships between feature patterns and label values. These
patterns are discovered through the training process.

3.      
Training: To train a model, it is provided with
labeled datasets. If its prediction has some discrepancy from the actual
findings, the same is provided to the model for improvement of its predictions.
This process is repeated for each data item in the data set.

4.     
Evaluating: To evaluate a trained model, it is given
an input consisting of unlabeled features whose labels are known by the
evaluator. Then the model outputs some values based on the features. The closer
it predicts the labels, the better is the model.

# Types of ML

Machine
Learning is a subset of Artificial Intelligence (AI). It includes the process
of taking data as input and to obtain meaningful conclusions by analysis. The
type of analysis done helps classify ML into various types:

1.       
Supervised learning: In this type the user trains the
model by giving it a set of labels. The model then uses these to make
predictions.

a.      
Regression: Where the aim is to output a numeric value

                                                        
i.           
Linear: Linear methods are used for obtaining a value.

                                                      
ii.           
Polynomial: Some non linear methods are used.

                                                    
iii.           
Logistic regression: Returns True or False.

                                                    
iv.           
Lasso,ridge and other regression algorithms.

b.      
Classification: The output is a set of classes divided
on basis of some criteria as per the data provided to the model.

                                                        
i.           
Binary: For yes/no classifications

                                                      
ii.           
Multiclass: For more complex classifications which
have multiple groups

                                                    
iii.           
Multilabel: While multiclass only assigns one class to
one data item, Multilabel allows one data item to have multiple class
attributes.

c.      
Ensemble: In this process multiple models are used to
do the same task. The results are then combined to form the ideal model.

2.      
Semi supervised learning: The model is given a set of labels
 as per the method of supervised
learning, but it uses this data to work on a larger set where answers are not
provided and analysis is done by the model from ground up.

3.      
Self Supervised learning: It is the bridge between
supervised and unsupervised learning. Instead of using given labels, the model
is trained to identify the labels itself and perform analysis on them.

4.     
Unsupervised learning: Here the data provided to the
model is unlabelled and the duty of the model is to arrange the data into a
meaningful manner. It identififes patterns and links between data items. The
user may afterwards label the data as required.

a.      
Clustering: Arrange data into clusters based on
similarity.

i.                    
Exclusive(hard clustering): In this grouping 1 data
point can only exist in 1 cluster. K-means clustering is a common example where
points are assigned to K groups.

ii.                  
Hierarchal: 

·        
Agglomerative: Here the data points existing in groups
are arranged in a heirarchial order by merging together iteratively on basis of
some similarity until a cluster is achieved.

·        
Divisive: A single cluster is divided based on
differences between the data points within the cluster.  

iii.                
Overlapping{soft clustering):  Here, 1 data point can be assigned to
multiple clusters or groups. Eg. Fuzzy K-means.

iv.                
Probabilistic : Here the  data points are clustered on the basis of the
likelihood of thembelonging to a particular distribution. E.g Gaussian Mixture
Models which have multiple propbability distribution functions.      

b.      
Dimensionality 
Reduction: Reduce the Complexity of the dataset by keeping relevant data
and tossing the unneeded data without altering any form of analysis. Eg. SVD,PCA,
AutoEncoders.

5.      
Reinforcement learning: In this type of learning the
model is provided rewards for making progress and following preset instructions
and is punished for doing things that oppose the working of the task at hand.
The model tries to figure out the method that requires the minimum amount of
punishment while maximizing the rewards.

6.      
Generative AI: These are models that take user input
and generate required outputs. The inputs and outputs can be texts, images,
videos, audios, codes, files etc. Generative ai models learn data from their
training dataset and recognize the patterns to generate outputs.

# Data Cleaning 

Data
Cleaning is the process of converting a dataset into a polished form which can
be used to train a model.

**STEPS:**

1.       
Import
pandas and numpy and read the given data

2.      
Sanity
check of data

3.      
Exploratory
data analysis

4.     
Empty
value handling

5.      
Outlier,
dupe and garbage value treatment

6.      
Normalisation

7.      
Data
encoding 

8. **Handling      Missing Values:**      

In
real-world datasets, it is quite common to find some values missing. This can
happen because of errors while collecting data, incomplete forms, or because a
particular value was simply not recorded. Missing values need to be handled
properly because they can affect the accuracy of the ML model.

There
are several ways to deal with missing values depending on the type of data and
the reason for the missing information.

- **Removing      rows:** If      only a small number of rows have missing values, those rows can sometimes      be removed. 
- **Removing      columns:** If      a column contains too many missing values and is not very useful, it may      be better to remove the entire column. 
- **Filling      with mean:**      For numerical data, missing values can be replaced with the average value      of that column. 
- **Filling      with median:**      The median can be used instead of the mean when the data contains extreme      values or outliers. 
- **Filling      with mode:** For      categorical data, the most frequently occurring value can be used to fill      missing entries. 
- **Forward      or backward filling:**      In time-based datasets, a previous or following value can sometimes be      used to fill a missing value. 

The
method should be chosen carefully because simply filling every missing value
with the same number may introduce errors into the dataset.

9. **Scaling      and Normalisation:**      

Different
features in a dataset can have very different ranges. For example, a person's
age may range from 0 to 100, while their income may range from thousands to
millions. If these values are directly given to certain ML algorithms, the
feature with the larger numerical range may have an unnecessarily large
influence.

**Scaling** is used to bring numerical
features into comparable ranges.

Two
common methods are:

- **Min-Max      Scaling:**      Converts values into a fixed range, usually between 0 and 1. 
- **Standardisation:** Transforms the values so      that they have a mean of 0 and a standard deviation of 1. 

Normalisation
is especially useful for algorithms that are sensitive to the scale of the
input data, such as K-Nearest Neighbours, K-Means and some gradient-based
algorithms.

10. **Parsing      Dates:** 

Dates
can often be difficult for a machine learning model to understand if they are
stored simply as text. For example, a value such as 16/09/2026 is meaningful to
a human, but the computer may initially treat it as just a string.

Therefore,
dates are usually converted into a proper date/time format. Once this is done,
useful information can be extracted from them, such as:

- Day      
- Month      
- Year      
- Day      of the week 
- Hour      
- Minute      

For
example, instead of using the complete date as a single value, we could create
separate features for the **month** or **day of the week**. This can help
the model identify patterns that depend on time.

11. **Character      Encoding:** 

Character
encoding determines how characters are stored and interpreted by a computer.
This becomes important when datasets contain different languages, special
symbols or characters such as é, ₹, or other Unicode characters.

Sometimes,
while importing a CSV file, an incorrect encoding can cause errors or turn the
original characters into unreadable symbols. This is why the correct character
encoding needs to be identified while loading the dataset.

Common
encodings include **UTF-8**, which is widely used because it can represent
characters from many different languages.

12. **Inconsistent      Data Entry:** 

Data
collected from different people or sources can often contain the same
information in different forms. For example, a column representing gender might
contain values such as Male, male, M, and MALE. Although these values may mean
the same thing to a human, a computer can treat them as different categories.

Similarly,
a country could be entered as India, india, or IND.

These
inconsistencies should be identified and corrected before training the model.
This can involve:

- Converting      text to a common case. 
- Removing      unnecessary spaces. 
- Correcting      spelling mistakes. 
- Replacing      different versions of the same value with one standard value. 
- Making      sure units are consistent throughout the dataset. 

This
step helps prevent the model from treating identical information as different
information.

13. **Removing      Duplicate Values:**      

Sometimes
the same record may accidentally appear more than once in a dataset. These
duplicate records can make certain observations appear more frequently than
they actually occur and may affect the model's learning.

Therefore,
duplicate rows should be identified and removed when they do not represent
separate observations.

For
example, if the same student's information has been entered twice by mistake,
keeping both entries could give that student twice as much influence on the
dataset.

14. **Handling      Outliers:** 

An
**outlier** is a value that is unusually far away from most of the other
values in a dataset. For example, if the ages in a dataset are mostly between
18 and 60 but one value is recorded as 250, that value may be an error.

Outliers
can be detected using methods such as:

- Box      plots 
- Interquartile      Range (IQR) 
- Z-scores      
- Statistical      analysis 

However,
an outlier should not automatically be deleted. Some unusual values may
actually represent real situations. For example, a very high transaction amount
may be unusual but still completely valid.

Therefore,
the reason behind an outlier should be considered before removing or changing
it.

15. **Checking      Data Types:** 

Every
column in a dataset has a particular data type, such as integer, floating-point
number, string or date. Sometimes a column may have been imported using the
wrong data type.

For
example, a column containing numbers might be stored as text because some
entries contain unnecessary symbols. Before using the dataset, the data types
should be checked and converted where necessary.

Correct
data types make the dataset easier to analyse and also prevent errors during
model training.

16. **Exploratory      Data Analysis (EDA):**      

After
cleaning the basic problems in the dataset, it is useful to explore the data
and understand what it actually contains. This process is known as **Exploratory
Data Analysis**.

EDA
can include:

- Finding      the minimum and maximum values. 
- Calculating      averages and other statistics. 
- Checking      how frequently different categories occur. 
- Studying      relationships between different features. 
- Creating      graphs and visualisations. 
- Looking      for unexpected patterns. 

EDA
is important because simply cleaning the data does not guarantee that we
understand it. By exploring the dataset, we can find patterns and problems that
might not be obvious just by looking at individual rows.

17. **Final      Data Validation:**      

Before
giving the cleaned dataset to a machine learning model, a final check should be
performed. This ensures that the data does not still contain obvious errors.

Some
things that can be checked include:

- Whether      important columns contain missing values. 
- Whether      the data types are correct. 
- Whether      duplicate records have been handled. 
- Whether      categorical values are consistent. 
- Whether      numerical values are within reasonable ranges. 
- Whether      the required features have been encoded correctly. 

Once
these checks are complete, the dataset can be considered ready for the next
stage of the machine learning process.

**Overall
Data Cleaning Workflow**

In
simple terms, the data cleaning process can be thought of as:

**Import
Data → Sanity Check → Explore Data →
Handle Missing Values → Remove Duplicates → Handle Outliers and Garbage Values → Normalise/Scale → Encode Data →
Parse Dates → Fix Inconsistent Entries → Validate Data →
Use for ML**

Data
cleaning may seem like a small part of machine learning, but it has a major
effect on the final model. A model can only learn properly from the information
it receives. Therefore, even a sophisticated ML algorithm can give poor results
if the data provided to it is incomplete, inconsistent or incorrect.