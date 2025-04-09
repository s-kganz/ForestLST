library(mgcv)
library(tidyverse)

df <- read_csv("ForestLST/data_working/westmort_tabular.csv")

train_years <- seq(2013, 2023, by=1)
valid_years <- seq(1997, 2004, by=1)
test_years  <- seq(2005, 2012, by=1)

df_train <- df %>%
  filter(time %in% train_years)

df_valid <- df %>%
  filter(time %in% valid_years)

df_test <- df %>%
  filter(time %in% test_years)

mod <- gam(
  mort_nextyear ~ s(mort_ewma) + s(prcp) + s(elev) + s(vp) + s(forest_ba) + s(fire) + s(vod),
  data=df_train
)

valid_predict <- predict(mod, newdata=df_valid)

cor(valid_predict, df_valid$mort_nextyear) ^ 2
mean((valid_predict - df_valid$mort_nextyear) ^ 2)

test_predict <- predict(mod, newdata=df_test)

cor(test_predict, df_test$mort_nextyear) ^ 2
mean((test_predict - df_test$mort_nextyear) ^ 2)
