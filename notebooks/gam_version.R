library(mgcv)
library(mgcViz)
library(pROC)
library(tidyverse)

tensors <- read_csv("data_working/preisler-tensors-rectangular.csv") %>%
  select(-`system:index`, -.geo) %>%
  mutate(pct_mortality_integer = pct_mortality > 0)

mygam <- gam(
  pct_mortality_integer ~ s(prcp1) + s(prcp2) + s(prcp3) + 
    s(prcp4) + s(rhost) + s(near) + s(fire) + s(winter_tmin) + s(latitude, longitude),
  data=tensors,
  family=binomial(link="logit")
)

yhat = predict(mygam, tensors, type='response')
yhat_classif = yhat > 0.5

confusion_mtx <- table(tensors$pct_mortality_integer, yhat_classif)

# Calculate performance metrics for comparison to other models
acc <- mean(yhat_classif == tensors$pct_mortality_integer)
precision <- confusion_mtx[2, 2] / sum(confusion_mtx[2, ])
recall <- confusion_mtx[2, 2] / sum(confusion_mtx[, 2])
auc <- roc(tensors$pct_mortality_integer, yhat)$auc

cat("Accuracy:", acc, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("AUC:", auc, "\n")

# recreate Fig 10
pred_risk_level <- round(yhat, digits=1)

ggplot() +
  geom_boxplot(aes(x=pred_risk_level, y=tensors$pct_mortality,
                   group=pred_risk_level)) +
  scale_y_continuous(trans=scales::pseudo_log_trans(base=10, sigma=5))

# Plot smooths
v <- getViz(mygam)
plot(v, allTerms=T)
