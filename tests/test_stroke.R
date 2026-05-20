# Ensure paths resolve from project root
if(basename(getwd()) == "tests") setwd("..")

library(testthat)
library(shiny)
library(caret)
library(ggplot2)
library(dplyr)

# ── Load app dependencies ─────────────────────────────────────────────────
model    <- readRDS("stroke_model.rds")
roc_data <- readRDS("roc_data.rds")
df       <- read.csv("stroke_data_cleaned.csv")

df$stroke_f <- factor(df$stroke, labels=c("No Stroke","Stroke"))
df$risk_group <- ifelse(df$age >= 60 & df$hypertension == 1, "High Risk",
                 ifelse(df$age >= 60 | df$hypertension == 1 |
                        df$heart_disease == 1, "Medium Risk", "Low Risk"))

COLOURS <- c("No Stroke"="#4db6ac", "Stroke"="#1a3a4a")

# ══════════════════════════════════════════════════════════════════════════
# 1. FILE LOADING TESTS
# ══════════════════════════════════════════════════════════════════════════
test_that("Required files load correctly", {
  expect_true(file.exists("stroke_model.rds"))
  expect_true(file.exists("roc_data.rds"))
  expect_true(file.exists("stroke_data_cleaned.csv"))
})

test_that("Model loads as correct class", {
  expect_true(inherits(model, "train"))
})

test_that("ROC data has correct structure", {
  expect_true(is.list(roc_data))
  expect_true(all(c("actual","prob_lr","prob_dt","prob_rf") %in% names(roc_data)))
})

test_that("Dataset loads with correct dimensions", {
  expect_equal(ncol(df) >= 20, TRUE)
  expect_gt(nrow(df), 5000)
})

# ══════════════════════════════════════════════════════════════════════════
# 2. DATA INTEGRITY TESTS
# ══════════════════════════════════════════════════════════════════════════
test_that("No missing values in cleaned dataset", {
  expect_equal(sum(is.na(df$bmi)), 0)
  expect_equal(sum(is.na(df$avg_glucose_level)), 0)
  expect_equal(sum(is.na(df$age)), 0)
  expect_equal(sum(is.na(df$stroke)), 0)
})

test_that("Stroke column is binary", {
  expect_true(all(df$stroke %in% c(0, 1)))
})

test_that("Age values are within valid range", {
  expect_true(all(df$age >= 0))
  expect_true(all(df$age <= 120))
})

test_that("BMI values are within valid range", {
  expect_true(all(df$bmi > 0))
  expect_true(all(df$bmi < 100))
})

test_that("Glucose values are within valid range", {
  expect_true(all(df$avg_glucose_level > 0))
  expect_true(all(df$avg_glucose_level < 300))
})

test_that("Engineered features exist", {
  expect_true("glucose_bmi"  %in% names(df))
  expect_true("age_glucose"  %in% names(df))
  expect_true("age_bmi"      %in% names(df))
  expect_true("hypert_age"   %in% names(df))
})

test_that("Engineered features are computed correctly", {
  expect_equal(
    df$glucose_bmi[1],
    df$avg_glucose_level[1] * df$bmi[1]
  )
  expect_equal(
    df$age_glucose[1],
    df$age[1] * df$avg_glucose_level[1]
  )
})

test_that("Risk groups are assigned correctly", {
  expect_true(all(df$risk_group %in% c("High Risk","Medium Risk","Low Risk")))
  # High risk must be age 60+ AND hypertension
  high_risk <- df[df$risk_group == "High Risk",]
  expect_true(all(high_risk$age >= 60))
  expect_true(all(high_risk$hypertension == 1))
})

# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL PREDICTION TESTS
# ══════════════════════════════════════════════════════════════════════════
make_patient <- function(age=45, glucose=100, bmi=28,
                          hypertension=0, heart_disease=0,
                          gender=0, ever_married=0,
                          residence=0, work_type="Private",
                          smoking="never smoked"){
  data.frame(
    age               = age,
    avg_glucose_level = glucose,
    bmi               = bmi,
    hypertension      = hypertension,
    heart_disease     = heart_disease,
    gender            = gender,
    ever_married      = ever_married,
    Residence_type    = residence,
    work_typechildren     = as.integer(work_type=="children"),
    work_typeGovt_job     = as.integer(work_type=="Govt_job"),
    work_typeNever_worked = as.integer(work_type=="Never_worked"),
    work_typePrivate      = as.integer(work_type=="Private"),
    work_typeSelf.employed= as.integer(work_type=="Self-employed"),
    smoking_statusformerly.smoked = as.integer(smoking=="formerly smoked"),
    smoking_statusnever.smoked    = as.integer(smoking=="never smoked"),
    smoking_statussmokes          = as.integer(smoking=="smokes"),
    smoking_statusUnknown         = as.integer(smoking=="Unknown"),
    glucose_outlier_flag = as.integer(glucose > 145),
    glucose_bmi  = glucose * bmi,
    age_glucose  = age * glucose,
    age_bmi      = age * bmi,
    hypert_age   = hypertension * age
  )
}

test_that("Model returns probability between 0 and 1", {
  patient <- make_patient()
  prob <- predict(model, patient, type="prob")[,"Stroke"]
  expect_gte(prob, 0)
  expect_lte(prob, 1)
})

test_that("High risk patient scores higher than low risk patient", {
  high_risk <- make_patient(age=72, glucose=210, bmi=36,
                             hypertension=1, heart_disease=1,
                             smoking="formerly smoked")
  low_risk  <- make_patient(age=28, glucose=78, bmi=22,
                             hypertension=0, heart_disease=0,
                             smoking="never smoked")
  prob_high <- predict(model, high_risk, type="prob")[,"Stroke"]
  prob_low  <- predict(model, low_risk,  type="prob")[,"Stroke"]
  expect_gt(prob_high, prob_low)
})

test_that("High risk patient exceeds threshold 0.30", {
  high_risk <- make_patient(age=72, glucose=210, bmi=36,
                             hypertension=1, heart_disease=1,
                             smoking="formerly smoked")
  prob <- predict(model, high_risk, type="prob")[,"Stroke"]
  expect_gte(prob, 0.30)
})

test_that("Low risk patient falls below threshold 0.30", {
  low_risk <- make_patient(age=28, glucose=78, bmi=22,
                            hypertension=0, heart_disease=0,
                            smoking="never smoked")
  prob <- predict(model, low_risk, type="prob")[,"Stroke"]
  expect_lt(prob, 0.30)
})

test_that("Model prediction is deterministic", {
  patient <- make_patient(age=55, glucose=130, bmi=30)
  prob1 <- predict(model, patient, type="prob")[,"Stroke"]
  prob2 <- predict(model, patient, type="prob")[,"Stroke"]
  expect_equal(prob1, prob2)
})

# ══════════════════════════════════════════════════════════════════════════
# 4. COLOR TESTS
# ══════════════════════════════════════════════════════════════════════════
test_that("COLOURS vector is correctly defined", {
  expect_equal(length(COLOURS), 2)
  expect_true("No Stroke" %in% names(COLOURS))
  expect_true("Stroke" %in% names(COLOURS))
})

test_that("Color values are valid hex codes", {
  hex_pattern <- "^#[0-9A-Fa-f]{6}$"
  expect_match(COLOURS["No Stroke"], hex_pattern)
  expect_match(COLOURS["Stroke"],    hex_pattern)
  # Test all hardcoded colors used in app
  app_colors <- c("#1a3a4a","#2a9d8f","#4db6ac",
                  "#264653","#e76f51","#e9c46a")
  for(col in app_colors){
    expect_match(col, hex_pattern,
                 info=paste("Invalid hex color:", col))
  }
})

test_that("Colors are distinguishable between stroke outcomes", {
  expect_false(COLOURS["No Stroke"] == COLOURS["Stroke"])
})

test_that("ggplot renders without color errors", {
  expect_no_error({
    p <- ggplot(df[1:100,], aes(x=age, fill=stroke_f)) +
      geom_histogram(bins=10) +
      scale_fill_manual(values=COLOURS)
    ggplot_build(p)
  })
})

test_that("All plot color scales build without error", {
  # Executive age plot colors
  expect_no_error({
    test_data <- data.frame(
      Group=c("Under 40","40-59","60+"),
      Rate=c(1,3,13)
    )
    p <- ggplot(test_data, aes(x=Group, y=Rate, fill=Group)) +
      geom_bar(stat="identity") +
      scale_fill_manual(values=c("Under 40"="#4db6ac",
                                  "40-59"="#2a9d8f",
                                  "60+"="#1a3a4a"))
    ggplot_build(p)
  })

  # Risk group colors
  expect_no_error({
    test_data <- data.frame(
      Group=c("High Risk","Medium Risk","Low Risk"),
      Count=c(100,500,4000)
    )
    p <- ggplot(test_data, aes(x=Group, y=Count, fill=Group)) +
      geom_bar(stat="identity") +
      scale_fill_manual(values=c("High Risk"="#1a3a4a",
                                  "Medium Risk"="#2a9d8f",
                                  "Low Risk"="#4db6ac"))
    ggplot_build(p)
  })
})

# ══════════════════════════════════════════════════════════════════════════
# 5. PROBABILITY CONCEPT TESTS
# ══════════════════════════════════════════════════════════════════════════
test_that("Stroke prevalence is approximately 4.87%", {
  prevalence <- mean(df$stroke)
  expect_gt(prevalence, 0.04)
  expect_lt(prevalence, 0.06)
})

test_that("Hypertensive patients have higher stroke rate", {
  rate_hypert    <- mean(df$stroke[df$hypertension == 1])
  rate_no_hypert <- mean(df$stroke[df$hypertension == 0])
  expect_gt(rate_hypert, rate_no_hypert)
})

test_that("Elderly patients have higher stroke rate", {
  rate_elderly <- mean(df$stroke[df$age >= 60])
  rate_young   <- mean(df$stroke[df$age < 60])
  expect_gt(rate_elderly, rate_young)
})

test_that("High glucose patients have higher stroke rate", {
  rate_high   <- mean(df$stroke[df$glucose_outlier_flag == 1])
  rate_normal <- mean(df$stroke[df$glucose_outlier_flag == 0])
  expect_gt(rate_high, rate_normal)
})

# ══════════════════════════════════════════════════════════════════════════
# 6. ROC DATA TESTS
# ══════════════════════════════════════════════════════════════════════════
test_that("ROC probabilities are between 0 and 1", {
  expect_true(all(roc_data$prob_lr >= 0 & roc_data$prob_lr <= 1))
  expect_true(all(roc_data$prob_dt >= 0 & roc_data$prob_dt <= 1))
  expect_true(all(roc_data$prob_rf >= 0 & roc_data$prob_rf <= 1))
})

test_that("ROC actual labels are correct factor levels", {
  expect_true(all(roc_data$actual %in% c("No_Stroke","Stroke")))
})

test_that("All ROC vectors have equal length", {
  expect_equal(length(roc_data$actual),  length(roc_data$prob_lr))
  expect_equal(length(roc_data$prob_lr), length(roc_data$prob_dt))
  expect_equal(length(roc_data$prob_dt), length(roc_data$prob_rf))
})

# ══════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════════════
cat("\n========================================\n")
cat("STROKE APP TEST SUITE\n")
cat("========================================\n")
cat("All tests completed.\n")