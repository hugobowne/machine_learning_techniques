#################################################################################
###THE SET UP
#################################################################################

library( caret )
rm(list=ls(all=TRUE))
setwd("~/Documents/ML/")#SET YOUR WORKING DIRECTORY HERE
data <- read.csv("test_potus_by_county.csv", header = TRUE  )
load( "mymodelgbm.rda")
#################################################################################
###RUN MODEL AND WRITE PREDICTIONS TO .CSV
#################################################################################
predictions <- predict(gbmfit , data )

write.table(predictions , "predictions.csv" , row.names = FALSE , col.names = FALSE)
