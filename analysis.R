# ==============================================================================
# Data Cleaning, Preprocessing, and Preliminary Analysis in R
# Dataset: Titanic Passenger Data (Kaggle / Data Science Dojo mirror)
# ==============================================================================

suppressMessages({
  library(dplyr)
  library(ggplot2)
  library(corrplot)
})

setwd("/home/claude/titanic_project")
plot_dir <- "plots"

# ------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------
titanic <- read.csv("data/titanic.csv", stringsAsFactors = FALSE, na.strings = c("", "NA"))

cat("========== STRUCTURE OF RAW DATA ==========\n")
str(titanic)

cat("\n========== FIRST 6 ROWS ==========\n")
print(head(titanic))

cat("\n========== DIMENSIONS ==========\n")
print(dim(titanic))

# ------------------------------------------------------------------------
# 2. INITIAL MISSING VALUE ASSESSMENT
# ------------------------------------------------------------------------
cat("\n========== MISSING VALUES PER COLUMN (RAW) ==========\n")
missing_counts <- colSums(is.na(titanic))
missing_pct <- round(100 * missing_counts / nrow(titanic), 2)
missing_summary <- data.frame(Column = names(missing_counts),
                               Missing_Count = missing_counts,
                               Missing_Percent = missing_pct,
                               row.names = NULL)
missing_summary <- missing_summary[order(-missing_summary$Missing_Count), ]
print(missing_summary)

# Visualize missingness
png(file.path(plot_dir, "01_missing_values_bar.png"), width = 800, height = 500)
barplot(missing_summary$Missing_Count[missing_summary$Missing_Count > 0],
        names.arg = missing_summary$Column[missing_summary$Missing_Count > 0],
        col = "steelblue", main = "Missing Values by Column (Raw Data)",
        ylab = "Number of Missing Values", las = 1)
dev.off()

# ------------------------------------------------------------------------
# 3. DATA CLEANING
# ------------------------------------------------------------------------

titanic_clean <- titanic

# 3a. Drop the "Cabin" column: ~77% missing, too sparse to impute reliably.
#     Instead, engineer a binary flag capturing whether cabin info was recorded,
#     since that itself may correlate with passenger class/fare.
titanic_clean$Has_Cabin <- ifelse(is.na(titanic_clean$Cabin), 0, 1)
titanic_clean$Cabin <- NULL

# 3b. Impute "Age" (numeric, ~20% missing) using the MEDIAN grouped by Pclass and Sex,
#     which is more accurate than a single global median because age distributions
#     differ meaningfully across cabin class and gender in this dataset.
age_medians <- titanic_clean %>%
  group_by(Pclass, Sex) %>%
  summarise(median_age = median(Age, na.rm = TRUE), .groups = "drop")
print(age_medians)

titanic_clean <- titanic_clean %>%
  left_join(age_medians, by = c("Pclass", "Sex")) %>%
  mutate(Age = ifelse(is.na(Age), median_age, Age)) %>%
  select(-median_age)

# 3c. Impute "Embarked" (categorical, 2 missing) using the MODE, since only a
#     tiny fraction of records are affected and mode imputation is standard
#     practice for low-missingness categorical fields.
embarked_mode <- names(sort(table(titanic_clean$Embarked), decreasing = TRUE))[1]
titanic_clean$Embarked[is.na(titanic_clean$Embarked)] <- embarked_mode
cat("\nMode used to impute Embarked:", embarked_mode, "\n")

# 3d. Drop identifier / free-text columns not useful for numeric analysis
#     (Name, Ticket, PassengerId are unique identifiers with no analytic value here).
titanic_clean <- titanic_clean %>% select(-PassengerId, -Name, -Ticket)

cat("\n========== MISSING VALUES AFTER CLEANING ==========\n")
print(colSums(is.na(titanic_clean)))

# ------------------------------------------------------------------------
# 4. OUTLIER DETECTION (IQR method) on numeric variables: Age, Fare
# ------------------------------------------------------------------------
detect_outliers_iqr <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  which(x < lower | x > upper)
}

age_outliers  <- detect_outliers_iqr(titanic_clean$Age)
fare_outliers <- detect_outliers_iqr(titanic_clean$Fare)

cat("\n========== OUTLIER DETECTION (IQR RULE) ==========\n")
cat("Number of Age outliers:", length(age_outliers), "\n")
cat("Number of Fare outliers:", length(fare_outliers), "\n")

png(file.path(plot_dir, "02_outlier_boxplots.png"), width = 900, height = 500)
par(mfrow = c(1, 2))
boxplot(titanic_clean$Age, main = "Age - Outlier Check", col = "lightgreen", ylab = "Age")
boxplot(titanic_clean$Fare, main = "Fare - Outlier Check", col = "salmon", ylab = "Fare")
dev.off()

# Cap (winsorize) extreme Fare outliers at the 99th percentile rather than deleting rows,
# preserving sample size while limiting the influence of extreme values (e.g. Fare = 512).
fare_cap <- quantile(titanic_clean$Fare, 0.99, na.rm = TRUE)
titanic_clean$Fare_capped <- ifelse(titanic_clean$Fare > fare_cap, fare_cap, titanic_clean$Fare)
cat("\n99th percentile Fare cap applied at:", round(fare_cap, 2), "\n")

# ------------------------------------------------------------------------
# 5. NORMALIZATION / SCALING of numeric variables
# ------------------------------------------------------------------------
# Min-max normalization to [0,1] for Age and capped Fare, useful for downstream
# distance-based or gradient-based modeling.
min_max_norm <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))

titanic_clean$Age_norm  <- min_max_norm(titanic_clean$Age)
titanic_clean$Fare_norm <- min_max_norm(titanic_clean$Fare_capped)

cat("\n========== SUMMARY OF NORMALIZED COLUMNS ==========\n")
print(summary(titanic_clean[, c("Age_norm", "Fare_norm")]))

# ------------------------------------------------------------------------
# 6. ENCODING CATEGORICAL VARIABLES
# ------------------------------------------------------------------------
# 6a. Label/factor encoding for ordinal-ish / binary variables
titanic_clean$Sex_encoded <- ifelse(titanic_clean$Sex == "male", 1, 0)   # binary encoding
titanic_clean$Pclass <- factor(titanic_clean$Pclass, levels = c(1,2,3), ordered = TRUE)

# 6b. One-hot encoding for nominal "Embarked" (3 categories: C, Q, S)
embarked_dummies <- model.matrix(~ Embarked - 1, data = titanic_clean)
colnames(embarked_dummies) <- gsub("Embarked", "Embarked_", colnames(embarked_dummies))
titanic_clean <- cbind(titanic_clean, embarked_dummies)

cat("\n========== STRUCTURE AFTER CLEANING & ENCODING ==========\n")
str(titanic_clean)

write.csv(titanic_clean, "outputs/titanic_cleaned.csv", row.names = FALSE)

# ------------------------------------------------------------------------
# 7. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------------
cat("\n========== summary() OF CLEANED DATASET ==========\n")
print(summary(titanic_clean))

cat("\n========== SURVIVAL RATE OVERALL ==========\n")
print(round(prop.table(table(titanic_clean$Survived)) * 100, 2))

cat("\n========== SURVIVAL RATE BY SEX ==========\n")
print(round(prop.table(table(titanic_clean$Sex, titanic_clean$Survived), margin = 1) * 100, 2))

cat("\n========== SURVIVAL RATE BY PCLASS ==========\n")
print(round(prop.table(table(titanic_clean$Pclass, titanic_clean$Survived), margin = 1) * 100, 2))

# --- Visualization 1: Survival counts by sex ---
p1 <- ggplot(titanic_clean, aes(x = Sex, fill = factor(Survived))) +
  geom_bar(position = "dodge") +
  labs(title = "Survival Counts by Sex", x = "Sex", y = "Count", fill = "Survived") +
  theme_minimal()
ggsave(file.path(plot_dir, "03_survival_by_sex.png"), p1, width = 7, height = 5)

# --- Visualization 2: Survival by passenger class ---
p2 <- ggplot(titanic_clean, aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(position = "fill") +
  labs(title = "Survival Proportion by Passenger Class", x = "Pclass", y = "Proportion", fill = "Survived") +
  theme_minimal()
ggsave(file.path(plot_dir, "04_survival_by_pclass.png"), p2, width = 7, height = 5)

# --- Visualization 3: Age distribution ---
p3 <- ggplot(titanic_clean, aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "darkcyan", color = "white") +
  labs(title = "Age Distribution (Post-Imputation)", x = "Age", y = "Frequency") +
  theme_minimal()
ggsave(file.path(plot_dir, "05_age_distribution.png"), p3, width = 7, height = 5)

# --- Visualization 4: Fare distribution (capped) ---
p4 <- ggplot(titanic_clean, aes(x = Fare_capped)) +
  geom_histogram(binwidth = 10, fill = "coral", color = "white") +
  labs(title = "Fare Distribution (Capped at 99th Percentile)", x = "Fare", y = "Frequency") +
  theme_minimal()
ggsave(file.path(plot_dir, "06_fare_distribution.png"), p4, width = 7, height = 5)

# --- Visualization 5: Age vs Fare scatter, colored by survival ---
p5 <- ggplot(titanic_clean, aes(x = Age, y = Fare_capped, color = factor(Survived))) +
  geom_point(alpha = 0.6) +
  labs(title = "Age vs Fare by Survival", x = "Age", y = "Fare (capped)", color = "Survived") +
  theme_minimal()
ggsave(file.path(plot_dir, "07_age_vs_fare.png"), p5, width = 7, height = 5)

# --- Correlation matrix of numeric variables ---
numeric_vars <- titanic_clean %>%
  select(Survived, Age, SibSp, Parch, Fare_capped, Has_Cabin, Sex_encoded) %>%
  mutate(across(everything(), as.numeric))

cor_matrix <- cor(numeric_vars, use = "complete.obs")
cat("\n========== CORRELATION MATRIX ==========\n")
print(round(cor_matrix, 2))

png(file.path(plot_dir, "08_correlation_matrix.png"), width = 700, height = 700)
corrplot(cor_matrix, method = "color", type = "upper", addCoef.col = "black",
         tl.col = "black", tl.srt = 45, number.cex = 0.8,
         title = "Correlation Matrix of Numeric Variables", mar = c(0,0,2,0))
dev.off()

# ------------------------------------------------------------------------
# 8. KEY INSIGHTS (printed for documentation)
# ------------------------------------------------------------------------
cat("\n========== KEY INSIGHTS ==========\n")
cat("1. Overall survival rate was", round(mean(titanic_clean$Survived)*100,1), "%\n")
female_rate <- round(mean(titanic_clean$Survived[titanic_clean$Sex=="female"])*100,1)
male_rate <- round(mean(titanic_clean$Survived[titanic_clean$Sex=="male"])*100,1)
cat("2. Female survival rate:", female_rate, "% vs Male survival rate:", male_rate, "%\n")
class1_rate <- round(mean(titanic_clean$Survived[titanic_clean$Pclass==1])*100,1)
class3_rate <- round(mean(titanic_clean$Survived[titanic_clean$Pclass==3])*100,1)
cat("3. 1st class survival rate:", class1_rate, "% vs 3rd class survival rate:", class3_rate, "%\n")
cat("4. Fare and Pclass show a meaningful negative relationship (higher fare -> lower class number)\n")
cat("5. Having a recorded Cabin (Has_Cabin=1) correlates with higher survival and fare, consistent with cabin location being a proxy for deck/class\n")

cat("\n===== SCRIPT COMPLETE =====\n")
