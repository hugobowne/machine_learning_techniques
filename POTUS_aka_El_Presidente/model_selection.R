#################################################################################
###THE SET UP
#################################################################################
library( glmnet )
library( ggplot2 )
library( caret )
library( kernlab )
library( klaR )
library(doMC)


###HERE I TAKE ADVANTAGE OF MULTITHREADING
###Using multithreading with my Dual Core 2.8 GHz Intel Core i7 processor,
###the code below takes ~6minutes to run
nc <- detectCores()
registerDoMC(cores = nc)


rm(list=ls(all=TRUE))
setwd("~/Documents/ML/")#SET YOUR WORKING DIRECTORY HERE
data <- read.csv("train_potus_by_county.csv", header = TRUE  )


#################################################################################
###INITIAL DATA DIVE
#################################################################################
#HISTOGRAM OF RESPONSE VARIABLE TO CHEK FOR CLASS IMBALANCE
q <- ggplot( data , aes(x=Winner))
q + geom_histogram() ##note a class imbalance!
##CHECK FOR FEATURES WITH NEAR-ZERO VARIANCE (MAY THROW OFF MODELS)
nzv <- nearZeroVar( data , saveMetrics=TRUE ) 
#View(nzv) ##no variables with near-zero variance
#VISUALIZE ALL VARIABLES AND THEIR RELATIONSHIPS
#ggpairs( data ) #THIS FUNCTION IS COMPUTATIONALLY INTENSIVE AND NOT ESSENTIAL FOR WHAT FOLLOWS

#################################################################################
###FEATURE SELECTION: I USE LASSO REGRESSION TO SELECT THE MOST IMPORTANT 
###FEATURES IN DETERMINING THE WINNER
###(YOU COULD ALSO USE A NONLINEAR ALGORITHM, SUCH AS A RANDOM FOREST
###TO SELECT FEATURES: AN ADVANTAGE OF LASSO REGRESSION IS THAT IT
###SELECTS FEATURES AND TELLS YOU WHETHER THEY ARE +VELY OR -VELY
###CORRELATED WITH THE TARGET VARIABLE)
#################################################################################
###SETUP INPUTS TO MODEL
n <- length( data )
x<- as.matrix(data[,-n])
y <- as.matrix(data$Winner)
###RUN THE MODEL
cvfit = cv.glmnet(x, y,  family = "binomial", type.measure = "class",
                  nfolds = 20 , nlambda = 1000 , alpha = 1)
##VARIABLES WITH NONZERO COOEFICIENTS ARE THE IMPORTANT VARIABLES
coef(cvfit$glmnet.fit,s=cvfit$lambda.1se) 

###KEEP IMPORTANT FEATURES AND RESPONSE VARIABLE
keep <- c("Median.age","X..BachelorsDeg.or.higher","Unemployment.rate",
          "Total.households","X..Owner.occupied.housing","X..Renter.occupied.housing",
          "Median.home.value","Population.growth", "Per.capita.income.growth",
          "Winner")
data <- data[,keep] ##KEEP ONLY THE MOST IMPORTANT FEATURES & RESPONSE VARAIBLES

#################################################################################
###IN WHICH I BUILD A NUMBER OF MODELS TO PREDICT THE RESPONSE VARIABLE
###I TRY LOGISTIC REGRESSION, SVMs, NEURAL NETWORKS, RANDOM FORESTS,
###GENERALIZED BOOSTED MODELS AND NAIVE BAYES.
###NOTE: PREPROCESSING OCCURS WITHIN EACH TRAINING METHOD.
#################################################################################


###DETAILS OF MODEL TRAINING (REPEATED 10-FOLD CROSS VALIDATION)
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  #classProbs = TRUE,
  ## repeated ten times
  repeats = 10)

###I DEFINE THE PREPROCESSING THAT I'LL PERFORM IN EACH MODEL FITTING
preProc = c("center", "scale","pca") ##centre & scale data, pca on predictor variables
tL = 5 #number of levels for each tuning parameter in training: you could do much wider and 
        #more rigorous tuning by choosing the model-dependent parameter values. Do this and
        #your models will perform better!!


# Start the clock!
ptm <- proc.time()
###LOGISTIC REGRESSION (AS A PARTICULAR "GENERAL LINEAR MODEL")
lrfit <- train( Winner ~. , data = data , method = "glm", family = binomial,
                trControl = fitControl, preProc , 
                tuneLength =tL)

###SUPPORT VECTOR MACHINE (RADIAL BASIS KERNEL)
svmfit <- train( Winner ~. , data = data , method = 'svmRadial',
                 trControl = fitControl, preProc  , 
                 tuneLength = tL)

###NEURAL NETWORK

nnetfit <- train( Winner ~. , data = data , method = "nnet",
                  trControl = fitControl, preProc)


###RANDOM FOREST

rffit <- train( Winner ~. , data = data , method = "rf",
                trControl = fitControl, preProc)


###GENERALIZED BOOSTED MODEL

gbmfit <- train( Winner ~. , data = data , method = "gbm", 
                 trControl = fitControl, preProc)

###NAIVE BAYES

nbfit <- train( Winner ~. , data = data , method = "nb", 
                 trControl = fitControl, preProc)
# Stop the clock
proc.time() - ptm
#################################################################################
###COMPARE ALL MODELS
#################################################################################
####

resamps <- resamples(list(nnet = nnetfit , gbm = gbmfit , lr = lrfit,
                          svm = svmfit , rf = rffit , nb = nbfit))
summary( resamps )
###GBM HAS THE HIGHEST MEAN ACCURACY

#################################################################################
###PRODUCE OUTPUTS
#################################################################################


###SAVE BEST MODEL TO THE FILESYSTEM
save(gbmfit , file = "mymodelgbm.rda")

###LOG DATA ABOUT EXPECTED PERFORMANCE OF MODEL
sink(file="performance.txt") 
gbmfit
sink(NULL) 




#################################################################################
###HERE BELOW I INCLUDE SOME PREPROCESSING CODE THAT CHECKS FOR FEATURES THAT 
### REMOVES HIGHLY CORRELATED FEATURES AND LOOKS FOR COLLINEARITY.
###THIS PREPROCESSING DID NOT IMPROVE MODEL PERFORMANCE
##SO I DID NOT INCLUDE IT IN THE ABOVE CODE.
#################################################################################

# ###remove correlated variables
# dummies <- dummyVars( ~ ., data )
# df <- predict(dummies, newdata = data)
# da <- data.frame(df)
# descrCor <-  cor( da )
# #summary(descrCor[upper.tri(descrCor)])
# highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
# filteredDescr <- da[,-highlyCorDescr]
# #descrCor2 <- cor(filteredDescr)
# #summary(descrCor2[upper.tri(descrCor2)])
# filteredDescr$Winner.Barack.Obama <- NULL
# filteredDescr$Winner <- data$Winner
# data <- filteredDescr
# ##find linear combos
# comboInfo <- findLinearCombos(data) #none