# Introduction

This report is for Coursera Practical Machine Learning Course’s Final Project.

The data is from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

Basically the data is collected from sensor device such as Jawbone Up, Nike FuelBand, and Fitbit which are attached on belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The objective of this project is to predict the manner in which the participants did the exercise. The variable which I am predicting is called “classe”.

Our outcome variable “classe” is a factor variable with 5 levels. For this dataset, “participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions:

exactly according to the specification (Class A)

throwing the elbows to the front (Class B)

lifting the dumbbell only halfway (Class C)

lowering the dumbbell only halfway (Class D)

throwing the hips to the front (Class E)

The report will touch on how the model is built, cross validation, out of sample error and predict the outcome for 20 different test subjects.
