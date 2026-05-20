suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(reshape2)
  library(corrplot)
  library(caret)
  library(randomForest)
  library(rpart)
  # rpart.plot not needed
  library(pROC)
  library(e1071)
})
set.seed(42)
setwd("E:/Projects/Masters/module_3/data_science/personal_projects/stroke_prediction")

# ── Load data ─────────────────────────────────────────────────────────────────
df <- read.csv("stroke_data.csv", header=TRUE)
cat("Rows:", nrow(df), "| Cols:", ncol(df), "\n")

# ── PART 2: Wrangling ─────────────────────────────────────────────────────────

# Missing values
cat("\nMissing counts:\n")
for(col in names(df)) cat(col, ":", sum(is.na(df[[col]])), "\n")

# Force bmi to numeric first - converts empty strings to proper NA
df$bmi <- as.numeric(df$bmi)

# Median imputation for bmi
df$bmi[is.na(df$bmi)] <- median(df$bmi, na.rm=TRUE)

# Check gender values
table(df$gender)

df <- df[df$gender != "Other", ]
cat("Rows after removing 'Other' gender:", nrow(df), "\n")

cat("Negative ages:", sum(df$age < 0), "\n")
cat("Zero BMI:", sum(df$bmi == 0), "\n")

# Outliers
Q1 <- quantile(df$avg_glucose_level, 0.25)
Q3 <- quantile(df$avg_glucose_level, 0.75)
IQR_val <- Q3 - Q1
upper <- Q3 + 1.5 * IQR_val
lower <- Q1 - 1.5 * IQR_val

cat("Upper fence:", upper, "\n")
cat("Lower fence:", lower, "\n")
cat("Glucose outliers flagged:", sum(df$avg_glucose_level > upper), "\n")

df$glucose_outlier_flag <- ifelse(df$avg_glucose_level > upper, 1, 0)

#Encoding
df$gender         <- ifelse(df$gender == "Male", 1, 0)
df$ever_married   <- ifelse(df$ever_married == "Yes", 1, 0)
df$Residence_type <- ifelse(df$Residence_type == "Urban", 1, 0)

#checking categories
# Check categories first so you know what columns will be created
cat("\nWork type categories:\n"); table(df$work_type)
cat("\nSmoking status categories:\n"); table(df$smoking_status)

# One-hot encoding
work_dummies    <- model.matrix(~ work_type - 1, data=df)
smoking_dummies <- model.matrix(~ smoking_status - 1, data=df)

#attaches the new columns to df
df <- cbind(df, work_dummies, smoking_dummies)

# Remove original text columns - no longer needed
df$work_type      <- NULL
df$smoking_status <- NULL

# Confirm new columns created
names(df)

# ── Feature Engineering ───────────────────────────────────────────────────
df$glucose_bmi <- df$avg_glucose_level * df$bmi
df$age_glucose <- df$age * df$avg_glucose_level
df$age_bmi     <- df$age * df$bmi
df$hypert_age  <- df$hypertension * df$age

# Confirm
cat("\nNew features added. Total columns:", ncol(df), "\n")
names(df)

# ── PART 3: EDA ───────────────────────────────────────────────────────────

# Outcome as factor for clean plot labels
df$stroke_f <- factor(df$stroke, labels=c("No Stroke", "Stroke"))

# ── Summary statistics ────────────────────────────────────────────────────
cat("\n--- Summary Statistics ---\n")
summary(df[, c("age", "avg_glucose_level", "bmi",
               "glucose_bmi", "age_glucose", "hypert_age")])

cat("\nStroke prevalence:", round(mean(df$stroke)*100, 2), "%\n")

# ── Plot 1: Class distribution (bar) ─────────────────────────────────────
ggplot(df, aes(x=stroke_f, fill=stroke_f)) +
  geom_bar() +
  geom_text(stat="count", aes(label=after_stat(count)), vjust=-0.5) +
  labs(title="Class Distribution: Stroke vs No Stroke",
       subtitle="Severe class imbalance — motivates SMOTE",
       x="Outcome", y="Count") +
  theme_minimal() +
  theme(legend.position="none")

# ── Plot 2: Age density by outcome ────────────────────────────────────────
ggplot(df, aes(x=age, fill=stroke_f)) +
  geom_density(alpha=0.5) +
  labs(title="Age Distribution by Stroke Outcome",
       subtitle="Stroke patients skew significantly older",
       x="Age", y="Density") +
  theme_minimal()

# ── Plot 3: Glucose boxplot by outcome ────────────────────────────────────
ggplot(df, aes(x=stroke_f, y=avg_glucose_level, fill=stroke_f)) +
  geom_boxplot(outlier.alpha=0.3) +
  labs(title="Glucose Levels by Stroke Outcome",
       x="Outcome", y="Average Glucose (mg/dL)") +
  theme_minimal() +
  theme(legend.position="none")

# ── Plot 4: BMI violin by outcome ─────────────────────────────────────────
ggplot(df, aes(x=stroke_f, y=bmi, fill=stroke_f)) +
  geom_violin(alpha=0.6) +
  geom_boxplot(width=0.1, fill="white") +
  labs(title="BMI Distribution by Stroke Outcome",
       x="Outcome", y="BMI (kg/m²)") +
  theme_minimal() +
  theme(legend.position="none")

# ── Plot 5: Age vs Glucose scatter coloured by outcome ────────────────────
ggplot(df, aes(x=age, y=avg_glucose_level, colour=stroke_f)) +
  geom_point(alpha=0.4, size=1.2) +
  labs(title="Age vs Glucose by Stroke Outcome",
       subtitle="Stroke cases cluster in high-age, high-glucose region",
       x="Age", y="Glucose (mg/dL)") +
  theme_minimal()

# ── Correlation analysis ──────────────────────────────────────────────────
num_cols <- c("age", "avg_glucose_level", "bmi",
              "glucose_bmi", "age_glucose", "age_bmi", "hypert_age",
              "hypertension", "heart_disease", "stroke")

cor_matrix <- cor(df[, num_cols], use="complete.obs")

# Correlations with stroke specifically - most important output
cat("\nCorrelations with stroke outcome (sorted):\n")
print(sort(cor_matrix[,"stroke"], decreasing=TRUE))

# Heatmap
corrplot(cor_matrix, method="color", type="upper",
         tl.cex=0.8, addCoef.col="black",
         number.cex=0.65,
         title="Correlation Matrix incl. Engineered Features",
         mar=c(0,0,1,0))

# ── PART 4: Probability & Statistical Analysis ────────────────────────────

# ── 1. Random Variables ───────────────────────────────────────────────────
cat("=== RANDOM VARIABLES ===\n")
cat("\nDiscrete Random Variables:\n")
cat("stroke, hypertension, heart_disease, gender, ever_married,\n")
cat("Residence_type, all work_type dummies, all smoking_status dummies\n")

cat("\nContinuous Random Variables:\n")
cat("age, bmi, avg_glucose_level\n")
cat("Engineered: glucose_bmi, age_glucose, age_bmi, hypert_age\n")

# ── 2. Probability Concepts ───────────────────────────────────────────────
cat("\n=== PROBABILITY CONCEPTS ===\n")

# Marginal probabilities
p_stroke       <- mean(df$stroke)
p_no_stroke    <- 1 - p_stroke
p_hypert       <- mean(df$hypertension)
p_heart        <- mean(df$heart_disease)

cat("\nP(Stroke):", round(p_stroke, 4), "\n")
cat("P(No Stroke):", round(p_no_stroke, 4), "\n")
cat("P(Hypertension):", round(p_hypert, 4), "\n")
cat("P(Heart Disease):", round(p_heart, 4), "\n")

# Conditional probabilities - clinically meaningful
# P(Stroke | Hypertension)
p_stroke_given_hypert <- mean(df$stroke[df$hypertension == 1])
p_stroke_given_no_hypert <- mean(df$stroke[df$hypertension == 0])
cat("\nP(Stroke | Hypertension):", round(p_stroke_given_hypert, 4), "\n")
cat("P(Stroke | No Hypertension):", round(p_stroke_given_no_hypert, 4), "\n")

# P(Stroke | High Glucose) - above upper fence we calculated earlier
p_stroke_given_high_glucose <- mean(df$stroke[df$glucose_outlier_flag == 1])
p_stroke_given_normal_glucose <- mean(df$stroke[df$glucose_outlier_flag == 0])
cat("\nP(Stroke | High Glucose):", round(p_stroke_given_high_glucose, 4), "\n")
cat("P(Stroke | Normal Glucose):", round(p_stroke_given_normal_glucose, 4), "\n")

# P(Stroke | Elderly) - age above 60
df$elderly <- ifelse(df$age >= 60, 1, 0)
p_stroke_given_elderly <- mean(df$stroke[df$elderly == 1])
p_stroke_given_young <- mean(df$stroke[df$elderly == 0])
cat("\nP(Stroke | Age >= 60):", round(p_stroke_given_elderly, 4), "\n")
cat("P(Stroke | Age < 60):", round(p_stroke_given_young, 4), "\n")

# Joint probability
# P(Stroke AND Hypertension)
p_stroke_and_hypert <- mean(df$stroke == 1 & df$hypertension == 1)
cat("\nP(Stroke AND Hypertension):", round(p_stroke_and_hypert, 4), "\n")

# Bayes theorem - P(Hypertension | Stroke)
# P(Hypert|Stroke) = P(Stroke|Hypert) * P(Hypert) / P(Stroke)
p_hypert_given_stroke <- (p_stroke_given_hypert * p_hypert) / p_stroke
cat("\nP(Hypertension | Stroke) via Bayes:", round(p_hypert_given_stroke, 4), "\n")
# Verify directly
cat("P(Hypertension | Stroke) direct:", 
    round(mean(df$hypertension[df$stroke == 1]), 4), "\n")

# ── 3. Distributions ──────────────────────────────────────────────────────
cat("\n=== DISTRIBUTIONS ===\n")

# stroke follows Binomial distribution
# X ~ Binomial(n, p) where p = 0.0487
cat("Stroke ~ Binomial(n=5109, p=", round(p_stroke,4), ")\n")
cat("Expected stroke cases:", round(nrow(df) * p_stroke), "\n")
cat("Actual stroke cases:", sum(df$stroke), "\n")

# Check normality of continuous variables
cat("\nShapiro-Wilk Normality Tests (sample of 500 due to test limits):\n")
set.seed(42)
samp <- sample(nrow(df), 500)
cat("age p-value:", shapiro.test(df$age[samp])$p.value, "\n")
cat("bmi p-value:", shapiro.test(df$bmi[samp])$p.value, "\n")
cat("avg_glucose_level p-value:", shapiro.test(df$avg_glucose_level[samp])$p.value, "\n")

# ── 4. Hypothesis Testing ─────────────────────────────────────────────────
cat("\n=== HYPOTHESIS TESTING ===\n")
alpha <- 0.05

# ── Test 1: Age ~ Stroke (one-tailed t-test) ──────────────────────────────
cat("\n--- Test 1: Age vs Stroke ---\n")
cat("H0: No difference in mean age between stroke and non-stroke patients\n")
cat("H1: Mean age is significantly higher in stroke patients\n")

age_stroke    <- df$age[df$stroke == 1]
age_no_stroke <- df$age[df$stroke == 0]

cat("Mean age - Stroke:", round(mean(age_stroke), 2), "\n")
cat("Mean age - No Stroke:", round(mean(age_no_stroke), 2), "\n")

t1 <- t.test(age_stroke, age_no_stroke, 
             alternative="greater",  # one-tailed: stroke group higher
             var.equal=FALSE)         # Welch's t-test
cat("t-statistic:", round(t1$statistic, 4), "\n")
cat("p-value:", round(t1$p.value, 6), "\n")
cat("Decision:", ifelse(t1$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0"), "\n")

# ── Test 2: age_bmi ~ Stroke (one-tailed t-test) ──────────────────────────
cat("\n--- Test 2: age_bmi vs Stroke ---\n")
cat("H0: No difference in mean age_bmi between stroke and non-stroke patients\n")
cat("H1: Mean age_bmi is significantly higher in stroke patients\n")

age_bmi_stroke    <- df$age_bmi[df$stroke == 1]
age_bmi_no_stroke <- df$age_bmi[df$stroke == 0]

cat("Mean age_bmi - Stroke:", round(mean(age_bmi_stroke), 2), "\n")
cat("Mean age_bmi - No Stroke:", round(mean(mean(age_bmi_no_stroke)), 2), "\n")

t2 <- t.test(age_bmi_stroke, age_bmi_no_stroke,
             alternative="greater",
             var.equal=FALSE)
cat("t-statistic:", round(t2$statistic, 4), "\n")
cat("p-value:", round(t2$p.value, 6), "\n")
cat("Decision:", ifelse(t2$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0"), "\n")

# ── Test 3: Hypertension ~ Stroke (chi-square) ────────────────────────────
cat("\n--- Test 3: Hypertension vs Stroke ---\n")
cat("H0: Hypertension is independent of stroke occurrence\n")
cat("H1: There is a significant relationship between hypertension and stroke\n")

hypert_table <- table(df$hypertension, df$stroke)
cat("\nContingency Table:\n")
print(hypert_table)

chi1 <- chisq.test(hypert_table)
cat("Chi-square statistic:", round(chi1$statistic, 4), "\n")
cat("p-value:", round(chi1$p.value, 6), "\n")
cat("Decision:", ifelse(chi1$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0"), "\n")

# ── Test 4: Smoking Status ~ Stroke (chi-square) ──────────────────────────
cat("\n--- Test 4: Smoking Status vs Stroke ---\n")
cat("H0: Smoking status is independent of stroke occurrence\n")
cat("H1: There is a significant relationship between smoking status and stroke\n")

# Need original smoking status - reconstruct from dummies
df$smoking_reconstructed <- ifelse(df$`smoking_statusformerly smoked` == 1, "formerly smoked",
                                   ifelse(df$`smoking_statusnever smoked` == 1, "never smoked",
                                          ifelse(df$`smoking_statussmokes` == 1, "smokes", "Unknown")))

smoking_table <- table(df$smoking_reconstructed, df$stroke)
cat("\nContingency Table:\n")
print(smoking_table)

chi2 <- chisq.test(smoking_table)
cat("Chi-square statistic:", round(chi2$statistic, 4), "\n")
cat("p-value:", round(chi2$p.value, 6), "\n")
cat("Decision:", ifelse(chi2$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0"), "\n")

# ── Summary table ─────────────────────────────────────────────────────────
cat("\n=== SUMMARY OF HYPOTHESIS TESTS ===\n")
cat(sprintf("%-35s %-12s %-10s %s\n", 
            "Test", "Statistic", "p-value", "Decision"))
cat(strrep("-", 75), "\n")
cat(sprintf("%-35s %-12s %-10s %s\n",
            "T-test: Age vs Stroke",
            round(t1$statistic, 4),
            round(t1$p.value, 6),
            ifelse(t1$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0")))
cat(sprintf("%-35s %-12s %-10s %s\n",
            "T-test: age_bmi vs Stroke",
            round(t2$statistic, 4),
            round(t2$p.value, 6),
            ifelse(t2$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0")))
cat(sprintf("%-35s %-12s %-10s %s\n",
            "Chi-sq: Hypertension vs Stroke",
            round(chi1$statistic, 4),
            round(chi1$p.value, 6),
            ifelse(chi1$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0")))
cat(sprintf("%-35s %-12s %-10s %s\n",
            "Chi-sq: Smoking vs Stroke",
            round(chi2$statistic, 4),
            round(chi2$p.value, 6),
            ifelse(chi2$p.value < alpha, "REJECT H0", "FAIL TO REJECT H0")))

# ── PART 5: Machine Learning ──────────────────────────────────────────────

# Install themis for SMOTE if not already installed
if(!require(themis)) install.packages("themis")
library(themis)

# ── Step 1: Prepare data ──────────────────────────────────────────────────
# Remove columns not needed for modelling
# id is just an identifier, stroke_f is a duplicate, 
# elderly and smoking_reconstructed are derived diagnostics
df_model <- df[, !names(df) %in% c("id", "stroke_f", 
                                   "elderly", 
                                   "smoking_reconstructed")]

# stroke must be a factor for classification
df_model$stroke <- factor(df_model$stroke, 
                          levels=c(0,1), 
                          labels=c("No_Stroke", "Stroke"))

cat("Class distribution before SMOTE:\n")
print(table(df_model$stroke))

# ── Step 2: Train/Test split 70/30 ───────────────────────────────────────
set.seed(42)
train_index <- createDataPartition(df_model$stroke, p=0.7, list=FALSE)
train_raw   <- df_model[train_index, ]
test_set    <- df_model[-train_index, ]

cat("\nTraining set size:", nrow(train_raw), "\n")
cat("Test set size:", nrow(test_set), "\n")
cat("\nTraining class distribution:\n")
print(table(train_raw$stroke))

# ── Step 3: Apply SMOTE to training set only ──────────────────────────────
# SMOTE balances the minority class by creating synthetic stroke cases
set.seed(42)

library(themis)

recipe_smote <- recipe(stroke ~ ., data=train_raw) %>%
  step_smote(stroke, over_ratio=1)

train_smote <- recipe_smote %>%
  prep() %>%
  bake(new_data=NULL)

cat("\nTraining class distribution AFTER SMOTE:\n")
print(table(train_smote$stroke))

# ── Step 4: Train control - 5 fold cross validation ──────────────────────
train_control <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,   # needed for probability predictions
  summaryFunction = twoClassSummary,  # gives ROC, Sens, Spec
  savePredictions = TRUE
)
# Clean column names - remove spaces, hyphens, special characters
names(train_smote) <- make.names(names(train_smote))
names(test_set)    <- make.names(names(test_set))

cat("Cleaned column names:\n")
print(names(train_smote))
# ── Step 5: Train models ──────────────────────────────────────────────────

# Model 1: Logistic Regression
cat("\nTraining Logistic Regression...\n")
set.seed(42)
model_lr <- train(stroke ~ .,
                  data      = train_smote,
                  method    = "glm",
                  family    = "binomial",
                  trControl = train_control,
                  metric    = "ROC")

# Model 2: Decision Tree
cat("Training Decision Tree...\n")
set.seed(42)
model_dt <- train(stroke ~ .,
                  data      = train_smote,
                  method    = "rpart",
                  trControl = train_control,
                  metric    = "ROC")

# Model 3: Random Forest
cat("Training Random Forest...\n")
set.seed(42)
model_rf <- train(stroke ~ .,
                  data      = train_smote,
                  method    = "rf",
                  trControl = train_control,
                  metric    = "ROC",
                  ntree     = 100)

# ── Step 6: Predict probabilities on test set ─────────────────────────────
prob_lr <- predict(model_lr, test_set, type="prob")[,"Stroke"]
prob_dt <- predict(model_dt, test_set, type="prob")[,"Stroke"]
prob_rf <- predict(model_rf, test_set, type="prob")[,"Stroke"]

# ── Step 7: ROC curves and AUC ───────────────────────────────────────────
roc_lr <- roc(test_set$stroke, prob_lr, levels=c("No_Stroke","Stroke"))
roc_dt <- roc(test_set$stroke, prob_dt, levels=c("No_Stroke","Stroke"))
roc_rf <- roc(test_set$stroke, prob_rf, levels=c("No_Stroke","Stroke"))

cat("\n=== AUC SCORES ===\n")
cat("Logistic Regression AUC:", round(auc(roc_lr), 4), "\n")
cat("Decision Tree AUC:      ", round(auc(roc_dt), 4), "\n")
cat("Random Forest AUC:      ", round(auc(roc_rf), 4), "\n")

# Plot ROC curves
plot(roc_lr, col="blue",  lwd=2, main="ROC Curves - All Models")
plot(roc_dt, col="red",   lwd=2, add=TRUE)
plot(roc_rf, col="green", lwd=2, add=TRUE)
legend("bottomright",
       legend=c(paste("Logistic Regression AUC=", round(auc(roc_lr),3)),
                paste("Decision Tree AUC=",       round(auc(roc_dt),3)),
                paste("Random Forest AUC=",       round(auc(roc_rf),3))),
       col=c("blue","red","green"), lwd=2)

# ── Step 8: Threshold tuning ──────────────────────────────────────────────
# Default threshold is 0.5 - we tune for best sensitivity
# We try thresholds from 0.1 to 0.5 and pick best sensitivity

tune_threshold <- function(probs, actual, thresholds=seq(0.1, 0.5, by=0.05)){
  results <- data.frame()
  for(t in thresholds){
    preds <- ifelse(probs >= t, "Stroke", "No_Stroke")
    preds <- factor(preds, levels=c("No_Stroke","Stroke"))
    cm    <- confusionMatrix(preds, actual, positive="Stroke")
    results <- rbind(results, data.frame(
      threshold   = t,
      sensitivity = round(cm$byClass["Sensitivity"], 4),
      specificity = round(cm$byClass["Specificity"], 4),
      accuracy    = round(cm$overall["Accuracy"], 4)
    ))
  }
  return(results)
}

cat("\n=== THRESHOLD TUNING ===\n")
cat("\nLogistic Regression:\n")
tune_lr <- tune_threshold(prob_lr, test_set$stroke)
print(tune_lr)

cat("\nDecision Tree:\n")
tune_dt <- tune_threshold(prob_dt, test_set$stroke)
print(tune_dt)

cat("\nRandom Forest:\n")
tune_rf <- tune_threshold(prob_rf, test_set$stroke)
print(tune_rf)

# ── Step 9: Final evaluation at tuned threshold ───────────────────────────
# Pick threshold with sensitivity >= 0.70 and highest specificity
# Adjust these based on your tuning results above

best_threshold_lr <- 0.30
best_threshold_dt <- 0.30
best_threshold_rf <- 0.30

evaluate_model <- function(probs, actual, threshold, model_name){
  preds <- ifelse(probs >= threshold, "Stroke", "No_Stroke")
  preds <- factor(preds, levels=c("No_Stroke","Stroke"))
  cm    <- confusionMatrix(preds, actual, positive="Stroke")
  cat("\n---", model_name, "at threshold", threshold, "---\n")
  cat("Confusion Matrix:\n")
  print(cm$table)
  cat("Accuracy:   ", round(cm$overall["Accuracy"], 4), "\n")
  cat("Sensitivity:", round(cm$byClass["Sensitivity"], 4), "\n")
  cat("Specificity:", round(cm$byClass["Specificity"], 4), "\n")
  cat("Precision:  ", round(cm$byClass["Precision"], 4), "\n")
  cat("F1 Score:   ", round(cm$byClass["F1"], 4), "\n")
  return(cm)
}

cat("\n=== FINAL MODEL EVALUATION ===\n")
cm_lr <- evaluate_model(prob_lr, test_set$stroke, best_threshold_lr, "Logistic Regression")
cm_dt <- evaluate_model(prob_dt, test_set$stroke, best_threshold_dt, "Decision Tree")
cm_rf <- evaluate_model(prob_rf, test_set$stroke, best_threshold_rf, "Random Forest")

# ── Step 10: Feature importance (Random Forest) ───────────────────────────
cat("\n=== FEATURE IMPORTANCE (Random Forest) ===\n")
importance_rf <- varImp(model_rf)
print(importance_rf)
plot(importance_rf, top=15, main="Top 15 Features - Random Forest")


# ── Final model summary ───────────────────────────────────────────────────
cat("=== FINAL MODEL SELECTION ===\n")
cat("\nWinner: Logistic Regression at threshold 0.30\n")
cat("\nJustification:\n")
cat("- Sensitivity 0.932 — catches 93 in every 100 real stroke patients\n")
cat("- Only 5 stroke patients missed out of 74 in test set\n")
cat("- Logistic Regression outperforms Random Forest on sensitivity\n")
cat("  without requiring aggressive threshold manipulation\n")
cat("- Coefficients are interpretable — clinically explainable\n")
cat("- Consistent with problem objective: early stroke risk screening\n")

# Logistic Regression coefficients
cat("\n=== LOGISTIC REGRESSION COEFFICIENTS ===\n")
coef_lr <- summary(model_lr$finalModel)$coefficients
coef_df <- data.frame(
  Variable = rownames(coef_lr),
  Coefficient = round(coef_lr[,1], 4),
  OddsRatio = round(exp(coef_lr[,1]), 4),
  PValue = round(coef_lr[,4], 4)
)
coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing=TRUE),]
print(coef_df)

# Final confusion matrix visual for winner
cat("\n=== WINNER: LOGISTIC REGRESSION CONFUSION MATRIX ===\n")
final_preds <- ifelse(prob_lr >= 0.30, "Stroke", "No_Stroke")
final_preds <- factor(final_preds, levels=c("No_Stroke","Stroke"))
final_cm    <- confusionMatrix(final_preds, test_set$stroke, positive="Stroke")
print(final_cm)

# ── PART 6: Data-Driven Decision Making ──────────────────────────────────

cat("=== RECOMMENDATIONS ===\n")
cat("\n1. HYPERTENSION FLAGGING\n")
cat("   P(Stroke|Hypertension) = 0.1325 vs 0.0397 without\n")
cat("   Action: All hypertensive patients flagged for immediate stroke risk review\n")

cat("\n2. AGE-BASED SCREENING\n")
cat("   P(Stroke|Age>=60) = 0.1315 vs 0.0182 under 60 — 7.2x multiplier\n")
cat("   Each additional year increases stroke odds by 11%\n")
cat("   Action: Annual mandatory stroke screening for all patients aged 60+\n")

cat("\n3. SMOKING HISTORY\n")
cat("   Chi-square p=0.000002 — smoking status significantly related to stroke\n")
cat("   Formerly smoked group had highest stroke count (70 cases)\n")
cat("   Action: Smoking history collected at every consultation for triage\n")

cat("\n=== RISK TRIAGE STRATEGY ===\n")
cat("High Risk:   Age>=60 AND Hypertension — immediate referral\n")
cat("Medium Risk: Age>=60 OR Hypertension OR Heart Disease — enhanced monitoring\n")
cat("Low Risk:    Age<60, no hypertension, no heart disease — annual screening\n")

cat("\n=== ALTERNATIVES EVALUATED ===\n")
cat("1. Random Forest considered but rejected — lower AUC (0.8064 vs 0.8274)\n")
cat("   required threshold 0.10 to match sensitivity, less stable\n")
cat("2. Without SMOTE — model would predict No Stroke for all patients\n")
cat("   achieving 95% accuracy but 0% sensitivity — clinically useless\n")
cat("3. Simple scoring system feasible for resource-limited settings\n")
cat("   using age + hypertension + glucose as three-variable triage tool\n")

cat("\n=== PROJECT SUMMARY ===\n")
cat("Dataset:     5,109 patients, 4.87% stroke prevalence\n")
cat("Best Model:  Logistic Regression, AUC 0.8274\n")
cat("Threshold:   0.30 (tuned for sensitivity)\n")
cat("Sensitivity: 0.932 — 93 in 100 stroke patients correctly identified\n")
cat("Top Features: age, age_glucose, age_bmi (engineered features validated)\n")
cat("Key Finding: Age>=60 with hypertension is highest risk group\n")

# ── Save cleaned dataset with engineered features ─────────────────────────
write.csv(df, "stroke_data_cleaned.csv", row.names=FALSE)
cat("\nCleaned dataset saved to stroke_data_cleaned.csv\n")
cat("Columns:", ncol(df), "| Rows:", nrow(df), "\n")

# ── Save model for Shiny app ──────────────────────────────────────────────
saveRDS(model_lr, "stroke_model.rds")
cat("Model saved to stroke_model.rds\n")

# Verify it loads correctly
test_load <- readRDS("stroke_model.rds")
cat("Model verified — class:", class(test_load), "\n")

saveRDS(test_set, "test_set.rds")

# Save all probabilities in one object
roc_data <- list(
  actual   = test_set$stroke,
  prob_lr  = prob_lr,
  prob_dt  = prob_dt,
  prob_rf  = prob_rf
)
saveRDS(roc_data, "roc_data.rds")

