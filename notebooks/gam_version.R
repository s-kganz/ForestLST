library(mgcv)
library(mgcViz)
library(pROC)
library(tidyverse)

tensors <- read_csv("data_working/preisler_dim_allyears.csv") %>%
  mutate(pct_mortality_integer = pct_mortality > 0)

mygam <- gam(
  pct_mortality ~ s(prcp1) + s(prcp2) + s(prcp3) + 
    s(prcp4) + s(rhost) + s(near) + s(fire) + s(winter_tmin) + s(latitude, longitude),
  data=tensors,
  family=binomial(link="logit")
)

# Accuracy metrics
yhat = predict(mygam, tensors, type='response')

r2 <- cor(yhat, tensors$pct_mortality)^2
rmse <- sqrt(mean((yhat - tensors$pct_mortality)^2))

# recreate Fig 10
pred_risk_level <- round(yhat, digits=1)

ggplot() +
  geom_boxplot(aes(x=pred_risk_level, y=tensors$pct_mortality,
                   group=pred_risk_level)) +
  scale_y_continuous(trans=scales::pseudo_log_trans(base=10, sigma=5)) +
  scale_x_continuous(breaks=seq(0, 1, by=0.1)) +
  theme_bw() +
  labs(x="Predicted risk level",
       y="True proportion mortality")

# 1 to 1
ggplot() +
  geom_bin2d(aes(x=tensors$pct_mortality, y=yhat)) +
  scale_fill_viridis_c(trans="log",
                       breaks=10^c(0, 2, 4, 6),
                       labels=c("10^0", "10^2", "10^4", "10^6")) +
  labs(x="True proportion mortality",
       y="Predicted proportion mortality") +
  ggtitle("GAM, all data") +
  theme_bw()

# Plot smooths
v <- getViz(mygam)
plot(v, allTerms=T)
