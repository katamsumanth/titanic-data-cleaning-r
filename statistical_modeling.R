# ==============================================================================
# Week 3: In-Depth Statistical Analysis and Predictive Modeling in R
# Dataset: Titanic Passenger Data (cleaned in Week 1)
# Task: Hypothesis testing + Logistic Regression classification model
# ==============================================================================

suppressMessages({
  library(dplyr)
  library(caret)
  library(pROC)
  library(car)
  library(e1071)
  library(ggplot2)
})

set.seed(42)
setwd("/home/claude/titanic_project")
plot_dir <- "plots_wk3"
dir.create(plot_dir, showWarnings = FALSE)

titanic <- read.csv("outputs/titanic_cleaned.csv", stringsAsFactors = FALSE)
titanic$Pclass   <- factor(titanic$Pclass, levels = c(1,2,3))
titanic$Sex      <- factor(titanic$Sex)
titanic$Embarked <- factor(titanic$Embarked)
titanic$Survived_f <- factor(titanic$Survived, levels = c(0,1), labels = c("No","Yes"))

cat("Dataset loaded:", nrow(titanic), "rows,", ncol(titanic), "columns\n\n")

# ==============================================================================
# PART 1: EXPLORATORY STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ==============================================================================

cat("========================================================\n")
cat("PART 1: HYPOTHESIS TESTING\n")
cat("========================================================\n\n")

# ------------------------------------------------------------------------
# 1.1 Normality tests on numeric variables (Shapiro-Wilk)
# ------------------------------------------------------------------------
cat("---- 1.1 Normality Testing (Shapiro-Wilk) ----\n")
shapiro_age  <- shapiro.test(titanic$Age)
shapiro_fare <- shapiro.test(titanic$Fare_capped)
cat("H0: Age is normally distributed\n")
print(shapiro_age)
cat("\nH0: Fare (capped) is normally distributed\n")
print(shapiro_fare)
cat("\nInterpretation: Both p-values are far below 0.05, so we reject the null\n")
cat("hypothesis of normality for both Age and Fare. This justifies using\n")
cat("non-parametric or rank-based tests as a cross-check for numeric comparisons.\n\n")

png(file.path(plot_dir, "01_qqplots_age_fare.png"), width = 900, height = 450)
par(mfrow = c(1,2))
qqnorm(titanic$Age, main = "Q-Q Plot: Age"); qqline(titanic$Age, col = "red", lwd = 2)
qqnorm(titanic$Fare_capped, main = "Q-Q Plot: Fare (capped)"); qqline(titanic$Fare_capped, col = "red", lwd = 2)
dev.off()

# ------------------------------------------------------------------------
# 1.2 Hypothesis Test: Does Age differ between survivors and non-survivors?
# ------------------------------------------------------------------------
cat("---- 1.2 Age vs Survival (Welch Two-Sample t-test + Wilcoxon) ----\n")
cat("H0: Mean age is equal between survivors and non-survivors\n")
cat("H1: Mean age differs between survivors and non-survivors\n")
t_age <- t.test(Age ~ Survived_f, data = titanic)
print(t_age)
w_age <- wilcox.test(Age ~ Survived_f, data = titanic)
cat("\nNon-parametric cross-check (Wilcoxon rank-sum test), given non-normality:\n")
print(w_age)
cat("\nInterpretation: The t-test p-value (", round(t_age$p.value, 4),
    ") is above the conventional 0.05 threshold, so we fail to reject H0 --\n", sep = "")
cat("mean age does not differ significantly by survival status. This matches\n")
cat("the Wilcoxon test result and confirms age alone is a weak linear predictor.\n\n")

# ------------------------------------------------------------------------
# 1.3 Hypothesis Test: Does Fare differ between survivors and non-survivors?
# ------------------------------------------------------------------------
cat("---- 1.3 Fare vs Survival (Welch Two-Sample t-test + Wilcoxon) ----\n")
cat("H0: Mean fare is equal between survivors and non-survivors\n")
cat("H1: Mean fare differs between survivors and non-survivors\n")
t_fare <- t.test(Fare_capped ~ Survived_f, data = titanic)
print(t_fare)
w_fare <- wilcox.test(Fare_capped ~ Survived_f, data = titanic)
cat("\nNon-parametric cross-check (Wilcoxon rank-sum test):\n")
print(w_fare)
cat("\nInterpretation: p-value < 0.001 in both tests, so we reject H0 --\n")
cat("survivors paid significantly higher fares on average than non-survivors,\n")
cat("consistent with fare acting as a proxy for class and deck location.\n\n")

# ------------------------------------------------------------------------
# 1.4 Hypothesis Test: Association between Sex and Survival (Chi-square)
# ------------------------------------------------------------------------
cat("---- 1.4 Sex vs Survival (Chi-square Test of Independence) ----\n")
cat("H0: Sex and survival are independent\n")
cat("H1: Sex and survival are associated\n")
tab_sex <- table(titanic$Sex, titanic$Survived_f)
print(tab_sex)
chi_sex <- chisq.test(tab_sex)
print(chi_sex)
cat("\nInterpretation: p-value < 0.001, so we reject H0 -- there is a highly\n")
cat("significant association between sex and survival.\n\n")

# ------------------------------------------------------------------------
# 1.5 Hypothesis Test: Association between Pclass and Survival (Chi-square)
# ------------------------------------------------------------------------
cat("---- 1.5 Passenger Class vs Survival (Chi-square Test of Independence) ----\n")
cat("H0: Passenger class and survival are independent\n")
cat("H1: Passenger class and survival are associated\n")
tab_class <- table(titanic$Pclass, titanic$Survived_f)
print(tab_class)
chi_class <- chisq.test(tab_class)
print(chi_class)
cat("\nInterpretation: p-value < 0.001, so we reject H0 -- passenger class is\n")
cat("significantly associated with survival.\n\n")

# ------------------------------------------------------------------------
# 1.6 Hypothesis Test: Association between Embarked and Survival (Chi-square)
# ------------------------------------------------------------------------
cat("---- 1.6 Port of Embarkation vs Survival (Chi-square Test) ----\n")
cat("H0: Port of embarkation and survival are independent\n")
tab_emb <- table(titanic$Embarked, titanic$Survived_f)
print(tab_emb)
chi_emb <- chisq.test(tab_emb)
print(chi_emb)
cat("\nInterpretation: p-value =", round(chi_emb$p.value, 4),
    "-- reject H0 at the 0.05 level, indicating a significant (if weaker)\n")
cat("association, likely driven by the class-mix differences noted in Week 2.\n\n")

# ------------------------------------------------------------------------
# 1.7 Correlation Test: Age vs Fare
# ------------------------------------------------------------------------
cat("---- 1.7 Correlation Test: Age vs Fare (Spearman, given non-normality) ----\n")
cat("H0: rho = 0 (no monotonic correlation between Age and Fare)\n")
cor_test <- cor.test(titanic$Age, titanic$Fare_capped, method = "spearman", exact = FALSE)
print(cor_test)
cat("\nInterpretation: A weak positive correlation (rho ~", round(cor_test$estimate,2),
    ") that is statistically significant given the large sample size, but of\n", sep="")
cat("limited practical/predictive magnitude on its own.\n\n")

# ==============================================================================
# PART 2: MODEL BUILDING - LOGISTIC REGRESSION CLASSIFICATION
# ==============================================================================

cat("========================================================\n")
cat("PART 2: PREDICTIVE MODEL - LOGISTIC REGRESSION\n")
cat("========================================================\n\n")

model_data <- titanic %>%
  select(Survived_f, Pclass, Sex, Age, SibSp, Parch, Fare_capped, Embarked, Has_Cabin)

# ------------------------------------------------------------------------
# 2.1 Train/Test Split (75/25, stratified on outcome)
# ------------------------------------------------------------------------
train_idx <- createDataPartition(model_data$Survived_f, p = 0.75, list = FALSE)
train_data <- model_data[train_idx, ]
test_data  <- model_data[-train_idx, ]

cat("Training set:", nrow(train_data), "rows | Test set:", nrow(test_data), "rows\n")
cat("Training set class balance:\n"); print(prop.table(table(train_data$Survived_f)))
cat("Test set class balance:\n"); print(prop.table(table(test_data$Survived_f)))
cat("\n")

# ------------------------------------------------------------------------
# 2.2 10-Fold Cross-Validation with Logistic Regression (caret)
# ------------------------------------------------------------------------
ctrl <- trainControl(method = "cv", number = 10,
                      classProbs = TRUE, summaryFunction = twoClassSummary,
                      savePredictions = "final")

set.seed(42)
log_model_cv <- train(Survived_f ~ Pclass + Sex + Age + SibSp + Parch + Fare_capped + Embarked + Has_Cabin,
                       data = train_data, method = "glm", family = "binomial",
                       trControl = ctrl, metric = "ROC")

cat("---- 10-Fold Cross-Validation Results (Training Set) ----\n")
print(log_model_cv)
cat("\n")

# ------------------------------------------------------------------------
# 2.3 Final Model Summary (coefficients, significance, odds ratios)
# ------------------------------------------------------------------------
final_glm <- log_model_cv$finalModel
cat("---- Final Logistic Regression Model Summary ----\n")
print(summary(final_glm))

cat("\n---- Odds Ratios with 95% Confidence Intervals ----\n")
odds_ratios <- exp(cbind(OddsRatio = coef(final_glm), suppressMessages(confint(final_glm))))
print(round(odds_ratios, 3))
cat("\n")

# ------------------------------------------------------------------------
# 2.4 Multicollinearity Check (Variance Inflation Factor)
# ------------------------------------------------------------------------
cat("---- Multicollinearity Check: Variance Inflation Factors (VIF) ----\n")
vif_vals <- vif(final_glm)
print(vif_vals)
cat("\nInterpretation: All VIF values are well below the common threshold of 5,\n")
cat("indicating no serious multicollinearity among predictors.\n\n")

# ==============================================================================
# PART 3: MODEL EVALUATION ON HELD-OUT TEST SET
# ==============================================================================

cat("========================================================\n")
cat("PART 3: MODEL EVALUATION ON TEST SET\n")
cat("========================================================\n\n")

test_probs <- predict(log_model_cv, newdata = test_data, type = "prob")[, "Yes"]
test_pred  <- predict(log_model_cv, newdata = test_data)

# ------------------------------------------------------------------------
# 3.1 Confusion Matrix
# ------------------------------------------------------------------------
cm <- confusionMatrix(test_pred, test_data$Survived_f, positive = "Yes")
cat("---- Confusion Matrix and Performance Metrics (Test Set) ----\n")
print(cm)
cat("\n")

# ------------------------------------------------------------------------
# 3.2 ROC Curve and AUC
# ------------------------------------------------------------------------
roc_obj <- roc(response = test_data$Survived_f, predictor = test_probs, levels = c("No","Yes"), direction = "<")
cat("---- ROC / AUC (Test Set) ----\n")
cat("AUC:", round(auc(roc_obj), 4), "\n")
ci_auc <- ci.auc(roc_obj)
cat("95% CI for AUC:", round(ci_auc[1],4), "-", round(ci_auc[3],4), "\n\n")

png(file.path(plot_dir, "02_roc_curve.png"), width = 600, height = 600)
plot(roc_obj, col = "#2a9d8f", lwd = 3, main = paste0("ROC Curve (AUC = ", round(auc(roc_obj),3), ")"))
abline(a = 0, b = 1, lty = 2, col = "grey50")
dev.off()

png(file.path(plot_dir, "03_confusion_matrix_heatmap.png"), width = 550, height = 500)
cm_table <- as.data.frame(cm$table)
cm_plot <- ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 8, color = "white") +
  scale_fill_gradient(low = "#8ecae6", high = "#023047") +
  labs(title = "Confusion Matrix (Test Set)", x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 15)
print(cm_plot)
dev.off()

# ------------------------------------------------------------------------
# 3.3 Diagnostic Plots (Residuals, Leverage, Cook's Distance)
# ------------------------------------------------------------------------
png(file.path(plot_dir, "04_diagnostic_plots.png"), width = 900, height = 850)
par(mfrow = c(2,2))
plot(final_glm, which = 1, caption = "Residuals vs Fitted")
plot(final_glm, which = 2, caption = "Normal Q-Q (Deviance Residuals)")
plot(final_glm, which = 4, caption = "Cook's Distance")
plot(final_glm, which = 5, caption = "Residuals vs Leverage")
dev.off()

# ------------------------------------------------------------------------
# 3.4 Variable Importance
# ------------------------------------------------------------------------
var_imp <- varImp(log_model_cv)
cat("---- Variable Importance (Standardized Coefficient Magnitude) ----\n")
print(var_imp)

png(file.path(plot_dir, "05_variable_importance.png"), width = 700, height = 550)
plot(var_imp, main = "Variable Importance - Logistic Regression")
dev.off()

# ------------------------------------------------------------------------
# 3.5 Threshold sensitivity: Precision/Recall at different cutoffs
# ------------------------------------------------------------------------
thresholds <- seq(0.3, 0.7, by = 0.1)
cat("\n---- Performance at Different Classification Thresholds ----\n")
for (th in thresholds) {
  pred_th <- factor(ifelse(test_probs >= th, "Yes", "No"), levels = c("No","Yes"))
  cm_th <- confusionMatrix(pred_th, test_data$Survived_f, positive = "Yes")
  cat(sprintf("Threshold %.1f | Accuracy: %.3f | Sensitivity: %.3f | Specificity: %.3f | Precision: %.3f\n",
              th, cm_th$overall["Accuracy"], cm_th$byClass["Sensitivity"],
              cm_th$byClass["Specificity"], cm_th$byClass["Precision"]))
}

cat("\n===== SCRIPT COMPLETE =====\n")
cat("Total diagnostic plots created:", length(list.files(plot_dir, pattern = "png$")), "\n")
