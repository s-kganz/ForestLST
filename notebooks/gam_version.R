library(mgcv)
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

table(tensors$pct_mortality_integer, yhat_classif)

# binary cross entropy

y <- tensors$pct_mortality_integer

xent <- mean(-y * log(yhat) - (1 - y) * log(1 - yhat))
