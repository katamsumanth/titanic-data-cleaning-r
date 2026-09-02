# ==============================================================================
# Week 2: Dynamic and Informative Visualizations in R
# Dataset: Titanic Passenger Data (cleaned in Week 1)
# Libraries demonstrated: ggplot2, lattice, base R graphics
# ==============================================================================

suppressMessages({
  library(dplyr)
  library(ggplot2)
  library(lattice)
})

setwd("/home/claude/titanic_project")
plot_dir <- "plots_wk2"
dir.create(plot_dir, showWarnings = FALSE)

titanic <- read.csv("outputs/titanic_cleaned.csv", stringsAsFactors = FALSE)
titanic$Pclass <- factor(titanic$Pclass, levels = c(1,2,3), labels = c("1st Class","2nd Class","3rd Class"))
titanic$Survived_lbl <- factor(titanic$Survived, levels = c(0,1), labels = c("Did Not Survive","Survived"))
titanic$Sex <- factor(titanic$Sex)

cat("Loaded cleaned dataset:", nrow(titanic), "rows,", ncol(titanic), "columns\n")
str(titanic[, c("Survived","Pclass","Sex","Age","Fare_capped","Embarked")])

# ------------------------------------------------------------------------
# CHART 1 (Base R): Histogram of Age
# ------------------------------------------------------------------------
png(file.path(plot_dir, "01_base_histogram_age.png"), width = 800, height = 550)
hist(titanic$Age, breaks = 20, col = "steelblue", border = "white",
     main = "Distribution of Passenger Age (Base R)",
     xlab = "Age (years)", ylab = "Number of Passengers")
abline(v = median(titanic$Age), col = "red", lwd = 2, lty = 2)
legend("topright", legend = paste("Median Age =", round(median(titanic$Age),1)),
       col = "red", lty = 2, lwd = 2, bty = "n")
dev.off()

# ------------------------------------------------------------------------
# CHART 2 (Base R): Boxplot of Fare by Passenger Class
# ------------------------------------------------------------------------
png(file.path(plot_dir, "02_base_boxplot_fare_class.png"), width = 800, height = 550)
boxplot(Fare_capped ~ Pclass, data = titanic,
        col = c("#66c2a5","#fc8d62","#8da0cb"),
        main = "Fare Paid by Passenger Class (Base R)",
        xlab = "Passenger Class", ylab = "Fare (capped at 99th percentile)")
dev.off()

# ------------------------------------------------------------------------
# CHART 3 (ggplot2): Bar chart - passenger counts by class
# ------------------------------------------------------------------------
p3 <- ggplot(titanic, aes(x = Pclass, fill = Pclass)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4, size = 5) +
  labs(title = "Number of Passengers by Class", x = "Passenger Class", y = "Number of Passengers") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")
ggsave(file.path(plot_dir, "03_bar_passenger_counts.png"), p3, width = 7.5, height = 5.5)

# ------------------------------------------------------------------------
# CHART 4 (ggplot2): Stacked bar chart - survival by class
# ------------------------------------------------------------------------
p4 <- ggplot(titanic, aes(x = Pclass, fill = Survived_lbl)) +
  geom_bar(position = "stack") +
  scale_fill_manual(values = c("Did Not Survive" = "#e76f51", "Survived" = "#2a9d8f")) +
  labs(title = "Survival Outcomes by Passenger Class", x = "Passenger Class", y = "Number of Passengers", fill = "Outcome") +
  theme_minimal(base_size = 14)
ggsave(file.path(plot_dir, "04_stacked_bar_survival_class.png"), p4, width = 7.5, height = 5.5)

# ------------------------------------------------------------------------
# CHART 5 (ggplot2): Grouped bar chart - survival rate (%) by class and sex
# ------------------------------------------------------------------------
rate_df <- titanic %>%
  group_by(Pclass, Sex) %>%
  summarise(Survival_Rate = mean(Survived) * 100, .groups = "drop")

p5 <- ggplot(rate_df, aes(x = Pclass, y = Survival_Rate, fill = Sex)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  geom_text(aes(label = paste0(round(Survival_Rate,1),"%")),
            position = position_dodge(width = 0.7), vjust = -0.4, size = 4) +
  scale_fill_manual(values = c("female" = "#e07a9e", "male" = "#4a7fb5")) +
  labs(title = "Survival Rate (%) by Class and Sex", x = "Passenger Class", y = "Survival Rate (%)", fill = "Sex") +
  ylim(0, 105) +
  theme_minimal(base_size = 14)
ggsave(file.path(plot_dir, "05_grouped_bar_survival_rate.png"), p5, width = 7.5, height = 5.5)

# ------------------------------------------------------------------------
# CHART 6 (ggplot2): Density plot - Age distribution by survival outcome
# ------------------------------------------------------------------------
p6 <- ggplot(titanic, aes(x = Age, fill = Survived_lbl)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("Did Not Survive" = "#e76f51", "Survived" = "#2a9d8f")) +
  labs(title = "Age Distribution: Survivors vs Non-Survivors", x = "Age (years)", y = "Density", fill = "Outcome") +
  theme_minimal(base_size = 14)
ggsave(file.path(plot_dir, "06_density_age_survival.png"), p6, width = 7.5, height = 5.5)

# ------------------------------------------------------------------------
# CHART 7 (ggplot2): Scatter plot - Age vs Fare, colored by survival, faceted by class
# ------------------------------------------------------------------------
p7 <- ggplot(titanic, aes(x = Age, y = Fare_capped, color = Survived_lbl)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = c("Did Not Survive" = "#e76f51", "Survived" = "#2a9d8f")) +
  facet_wrap(~ Pclass) +
  labs(title = "Age vs Fare by Survival, Split by Class", x = "Age (years)", y = "Fare (capped)", color = "Outcome") +
  theme_minimal(base_size = 13)
ggsave(file.path(plot_dir, "07_scatter_age_fare_facet.png"), p7, width = 9, height = 5.5)

# ------------------------------------------------------------------------
# CHART 8 (ggplot2): Line chart - Survival rate trend across age groups
# ------------------------------------------------------------------------
titanic$Age_Group <- cut(titanic$Age,
                          breaks = c(0,10,20,30,40,50,60,70,80),
                          labels = c("0-10","11-20","21-30","31-40","41-50","51-60","61-70","71-80"),
                          include.lowest = TRUE)

age_trend <- titanic %>%
  group_by(Age_Group) %>%
  summarise(Survival_Rate = mean(Survived) * 100, n = n(), .groups = "drop")

cat("\nSurvival rate by age group:\n")
print(age_trend)

p8 <- ggplot(age_trend, aes(x = Age_Group, y = Survival_Rate, group = 1)) +
  geom_line(color = "#264653", linewidth = 1.2) +
  geom_point(color = "#e76f51", size = 3) +
  geom_text(aes(label = paste0(round(Survival_Rate,0),"%")), vjust = -1, size = 4) +
  labs(title = "Survival Rate Trend Across Age Groups", x = "Age Group", y = "Survival Rate (%)") +
  ylim(0, 100) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 0))
ggsave(file.path(plot_dir, "08_line_survival_by_age_group.png"), p8, width = 8, height = 5.5)

# ------------------------------------------------------------------------
# CHART 9 (ggplot2): Proportional bar chart - Embarkation port vs class mix
# ------------------------------------------------------------------------
p9 <- ggplot(titanic, aes(x = Embarked, fill = Pclass)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("1st Class" = "#264653", "2nd Class" = "#2a9d8f", "3rd Class" = "#e9c46a")) +
  labs(title = "Passenger Class Mix by Port of Embarkation", x = "Port of Embarkation", y = "Proportion", fill = "Class") +
  theme_minimal(base_size = 14)
ggsave(file.path(plot_dir, "09_bar_embarked_class_mix.png"), p9, width = 7.5, height = 5.5)

# ------------------------------------------------------------------------
# CHART 10 (Lattice): Panelled histogram of Age conditioned on Survival
# ------------------------------------------------------------------------
png(file.path(plot_dir, "10_lattice_histogram_age_by_survival.png"), width = 900, height = 500)
print(
  histogram(~ Age | Survived_lbl, data = titanic,
            layout = c(2,1), col = "darkorange", type = "count",
            main = "Age Distribution Panelled by Survival Outcome (Lattice)",
            xlab = "Age (years)", ylab = "Count")
)
dev.off()

# ------------------------------------------------------------------------
# CHART 11 (Lattice): Scatter plot of Age vs Fare conditioned on Class
# ------------------------------------------------------------------------
png(file.path(plot_dir, "11_lattice_scatter_age_fare_by_class.png"), width = 950, height = 450)
print(
  xyplot(Fare_capped ~ Age | Pclass, data = titanic,
         groups = Survived_lbl, auto.key = list(space = "right"),
         pch = 19, alpha = 0.6, layout = c(3,1),
         main = "Age vs Fare Conditioned on Passenger Class (Lattice)",
         xlab = "Age (years)", ylab = "Fare (capped)")
)
dev.off()

cat("\n===== ALL VISUALIZATIONS GENERATED SUCCESSFULLY =====\n")
cat("Total plots created:", length(list.files(plot_dir, pattern = "png$")), "\n")
