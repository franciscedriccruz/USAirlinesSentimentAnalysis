# Sentiment Analysis of US Airlines in 2015 

Sentiment Analysis is a branch of Natural Language Processing (NLP) that allows us to determine whether a statement is either "positive" or "negative". The purpose of this notebook is to develop a model to compute the sentiment of text information pertaining to the public opinion of US airlines in 2015 on social media (Twitter). The model used for employing text based analysis is logistic regression. This model will be used to answer the research question: "What can public opinion on Twitter tell us about the US airlines in 2015?"

## Data Cleaning
Before delving into exploratory testing and data modelling, we first need to clean the data. I cleaned two data sets - the first being an extracted generic tweets data set and another dataset containing US Airline tweets. The generic tweets dataset will be used to train a model to predict the sentiment of the US airline tweet. Data cleaning was performed by first importing the tweets into a dataframe and then parsing through each tweet while applying certain operations to clean the text. One way I determined that the data required cleaning is through the length of the text. Typically, tweets are restricted to only 140 characters, but the data seemed to contain texts larger than this amount. 



Looking at some raw tweets, it seems that they contained HTML elements that made the text longer. Moreover, these also contained mentions of the airline and other users. The data cleaning process that I used was first to remove these HTML entities by using the BeautifulSoup library. Second, I removed contractions to accomodate for informalities in the English language especially with contractions. Mentions were also removed. However, if they mentioned a specific airline it would be stored as a column in the dataframe. I also removed stop works excluding the word "not" as it may have a large impact in the overall sentiment of the tweet. The histogram below depicts the changes in tweet length and it seems more reasonable now. 

PICTURE 2

## Exploratory Testing
Going through the cleaned and raw data, it is clear that the associated airline for each tweet is either typically mentioned in the beginning of the tweet or mentioned in the hashtag. Hence, I stored these information in a new column. Additionally, each tweet was already assigned a sentiment value and negative reason. Hence, this problem is an example of a supervised machine learning problem. I did a quick count and saw the United was most frequently tweeted about, particularly in a negative way. 

PICTURE 3

Comparing the negative and positive tweets, there were more negative tweets associated with the US Airlines in 2015. Among all airlines, United and US Airways received the bulk of negative tweets in the provided dataset. Nevertheless, there were also some people who tweeted positive things about these airlines, but these are a very small number. It should be noted that some tweets may have been attributed to multiple airlines. For example, a tweet comparing United to Virgin America may be negative for United but positive for the latter airline. Despite this possibility, it is assumed that the subject of the tweet would be mentioned in the beginning of the tweet as indicated in the raw data set.

### Other Visualization Methods
Other ways to present data found in the tweets is through using a word cloud. This would help identify which words most frequently come up for positive (shown in green) and negative (shown in red) tweets for the US Airline tweets. 

PICTURE 4

PICTURE 5

Based on the word clouds, we can see positive tweets in the US airline data set contain feelings of thankfulness most likely due to great customer service. On the other hand, words that show up in the negative sentiment tweets in the US airline data contain mostly the words relating to customer service, flights (probably caused by delayed or late flights). It is interesting to see the word "thank" in the word cloud since it must have been used in a sarcastic tone in the tweet.

A stacked bar graph illustrating the negative reasons associated with each airline helps us understand the distribution of reasons why people are displeased with an associated airline. 

PICTURE 6 

Based on the stacked bar graph, most of the negative reasons associated with the negative sentiment tweets in the US airline are dealing with customer service related issues. United, American Air, and US Airways typicallly have the most complaints among the variety of negative reasons. Most of which are due to poor customer service.

## Model Preparation 
To help aid with the features of the model, I decided to use a CountVectorizer to find the most common words and use them as features for each tweet based on a generic tweet dataframe. The outcome of the fit and transform of the CountVectorizer will result in a list for each tweet where each column represents a word from the most frequent corpus generated beforehand. While other parameters could have been used such as the TFIDVectorizer which is similar to the CountVectorizer but with a normalization within the specific sentence, this would require a larger computational time and would yield very similar results. The resulting vectorized output will be used as a feature to train my model. 

## Model Implementation
Seeing as this is a sentiment analysis problem, a good model to use for a classification problem is the logistic regression model. To see how well this model performs, I first trained a model using a vectorized generic tweets data set to see if it could predict the sentiment correctly. I did this by first dividing the model into 70% training and 30% testing sets. The logistic regression model was created using the scikit learn library. The confusion matrix of the resulting model is shown below: 

PICTURE 7

Ideally, a good model is one whose diagonal components of the confusion matrix should be relatively high as these correspond to the model's accuracy of true positives and true negatives. The confusion matrix also complements the previously reported recall and precision values of the positive and negative tweets. Based on these results, the model seems to perform well and can be used for US Airline tweets. Going through the same process, the resulting confusion matrix indicates that the model tends to incorrectly identify certain tweets as positive where in fact, they were negative. 

PICTURE 8

### How well did the predictions match the sentiment labelled in the US Airline Data? 
The model trained from the generic tweets data set achieved an accuracy of 78.18% in estimating the sentiment of the airline tweets. Looking into the classification report and confusion matrix, it can be seen that the model can estimate negative sentiment tweets better than the positive sentiment. The model was also noted to incorrectly estimate negative tweets as positive tweets. This may be attributed with the decision threshold between the two binary classes as seen on the curvature and area of the ROC curve shown inside the notebook (ideally, we would want this curve to be closer to the upper left corner and have an area of 1). This shape is attributed to challenging decision thresholds from the features of the tweets. Nevertheless, this model achieved a relatively high accuracy of 78.18% accuracy and is suitable for sentiment analysis.

## Analysis and Results
The logistic regression model was quite accurate in predicting the sentiment of the generic tweets (accuracy of 78.76%). This model was then applied to the US airlines tweets and achieved an accuracy of 78.18%.

### "What can public opinion on Twitter tell us about US airlines in 2015?"
From the data cleaning process, we saw that most tweets in the US airline dataset contained mostly negative sentiment. We will first look into the predicted sentiment and compare it to the actual sentiment available in the data set with respect to each US airline.

PICTURE 9


Based on a comparison of the distribution of the negative sentiment tweets, the model was quite accurate in determining the negative sentiment tweets of the US airlines. Most of the negative airlines were associated with United and US airways. The difference between the predicted and actual sentiment was very small - the model underestimated the negative sentiment towards United and US airways but the distribution of tweets classified in terms of sentiment was quite accurate.

PICTURE 10 

Based on a comparison of the distribution of the positive sentiment tweets, the model was not as accurate in determining the positive sentiment tweets compared to the negative sentiment tweets of the US airlines. For example, the model slightly underestimated the positive sentiment of the tweets associated to jetblue, and southwest air.

### Brief Answer to the Research Question: 
Overall, public opinion was mostly negative towards the US airlines in 2015. A model was trained to distinguish positive or negative sentiment using a generic tweets data set. This model was then applied to the US airline tweets data set to determine its sentiment value. The model was able to achieve accurate results, with an accuracy score of 78.18%. It was also noted that the model underestimated the negative sentiment tweets in the data set compared to its actual value. The model occasionally incorrectly classifies tweets as positive where in fact, they were negative. This inaccuracies can be attributed to the corpus used as the features to the model. The model was trained initially on the generic tweets data set and this data set may contain different words (feautures) that are more prominent in the generic tweets dataset than the US airline dataset. Hence, select words that may bear positive or negative sentiment in the US airline tweets may not be fully utilized in the model used. If the model was trained one the US airlines tweets, the accuracy of the model may slightly increase. Furthermore, features in both negative and positive sentiment have a small extent of overlapping words with each other as shown in the word clouds. Hence, the decision threshold of the model may have difficulty determining its sentiment based on the currently selected features. Nevertheless, the conclusion and trend from the output of the model and the actual sentiment values in the data set were the same. Most people were not satisfied with their experience with United, USAirways, and SouthWest Air the most compared to other airlines. Common reasons for the negative sentiment tweets will be discussed in the following section under the multi-class classification section.

In descending order, the worst airline according to the number of negative tweets were:

United - viewed more negative than any other airline
USAirways - viewed second most negative
AmericanAir - third worst airline
SouthWest Air - fourth worst airline despite having some positive tweets
JetBlue - almost even split between positive and negative tweets
Virgin America - almost even split between positive and negative tweets
Delta - not enough information to draw conclusion but among the data, it was negative.

## Suggestions to improve model accuracy
One method to improve the accuracy of both models is to tune the hyperparameters. This suggestion may lead to slightly higher performance but is also dependent on the features used in the model. Additionally, another method would also be to engineer better features to be used in the model. One example could be the use of a normalized word frequency for each sentence (TFID Vectorizer). This would normalized the word count with respect to the total amount of words in a given sentence instead of relying on the number of occurences of each word. Aside from these, one can also opt for another classification model such as the use of a decision tree classification model (for the multi-class classification model) as some categories are similar with subtle differences with one another. The decision tree will allow for a multi-class hierarchy that will aid in determining issues that are similar such as customer service related issues and the flight logistics issue as shown in the sample tweet. Neural networks can also be used for multiclass classification using Keras. However, with the work done in this project, the use of a decision tree classification model seems the best fit as most negative reasons are similar to one another with specific and minute differences. The decision tree will contain all the different classes as leaves on the tree. The features of the model will be used to determine the direction of branching within the tree. This technique will allow for checking individual features that are prominent within certain classes. Unlike logistic regression with a sigmoid function depicting close and vague thresholds between classes, the decision tree will be able to have distinct decision thresholds to differentiate multiple classes, thus achieving higher performance in the multi-class classification problem.
