# Background
# Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


# install necessary libraries
library(caret)
library(gridExtra)

# Reading in raw data
trainingRaw <- read.csv("pml-training.csv")
testingRaw <- read.csv("pml-testing.csv")

# Cleaning up training data set
# Most of the data are not very useful with a lot of NAs so good idea to subset the data sets, reducing to 52 variables
trainingSub <- subset(trainingRaw, select = grep("^accel|^total|^roll|^pitch|^yaw|^magnet|^gyro", names(trainingRaw)))
testingSub <- subset(testingRaw, select = grep("^accel|^total|^roll|^pitch|^yaw|^magnet|^gyro", names(testingRaw)))
trainingSub$Classe = trainingRaw$classe
testingSub$problem.id = testingRaw$problem_id

# clear memory
rm(trainingRaw, testingRaw)


# Creating data partitions within the training data set so that we can perform cross validation
set.seed("999")
inTrain <- createDataPartition(y=trainingSub$Classe, p=0.7, list=FALSE)
training <- trainingSub[inTrain,]
testing <- trainingSub[-inTrain,]


# First use decision tree technique
modFit <- train(Classe ~., method="rpart", data=training)

# Analyse final model
print(modFit$finalModel)

# Plotting partition (using ? as rattle can't install on my box)
plot(modFit$finalModel, uniform = TRUE)
text(modFit$finalModel, use.n=TRUE, all=TRUE)

# Looks ok, let's see how it does with prediction
pred <- predict(modFit, testing)
confusionMatrix(testing$Classe, pred)



# Results are not good. Accuracy is under 50% which is worse than a coin toss! 

# Trying Naive Bayes approach
modFit2 <- train(Classe ~., method="nb", data=training)
print(modFit2)
pred2 <- predict(modFit2, testing)
confusionMatrix(testing$Classe, pred2)

# Accuracy is encouraging - 76%

# Trying Random Forest technique
modFit3 <- train(Classe ~., method="rf", data=training)
print(modFit3)

pred3 <- predict(modFit3, testing)
confusionMatrix(testing$Classe, pred3)

# Accuracy is good - 99%, specificity and sensitivity also good - 99%

# Most of the errors come with misclassifying C with D. 
# Let's look at which features are most important
varImp(modFit3)

# If we plot the 3 most important features against each other
plot1 <- qplot(roll_belt,yaw_belt,colour=Classe,data=testing)
plot2 <- qplot(roll_belt,pitch_forearm,colour=Classe,data=testing)
grid.arrange(plot1, plot2, ncol=2)

# As you can see, the 3 most important features do not discriminate C/D very well

# Checking In Sample error
inpred <- predict(modFit3, training)
confusionMatrix(training$Classe, inpred)

# As you can see, the in sample error is extremely low with accuracy of 100% and sensitivity and specificity at 100% for all 3 classes
# With the testing data, the out of sample error is not too bad ie. 99.6%. This will be our final model

answer <- predict(modFit3, testingSub)
print(answer)

# Conclusion
# The Random Forest approach was by far the superior technique over both Decision Tree (ie. rpart) and Naive Baysian approaches. 
# Of the 5 classes, C and D are the most difficult to differentiate but Random Forest method gives the best result



