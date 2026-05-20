library(shiny)
library(ggplot2)
library(dplyr)
library(reshape2)
library(pROC)
library(caret)
library(shinycssloaders)

# ── Load everything once at startup ──────────────────────────────────────
model    <- readRDS("stroke_model.rds")
roc_data <- readRDS("roc_data.rds")
df       <- read.csv("stroke_data_cleaned.csv")

df$stroke_f <- factor(df$stroke, labels=c("No Stroke","Stroke"))
df$risk_group <- ifelse(df$age >= 60 & df$hypertension == 1, "High Risk",
                        ifelse(df$age >= 60 | df$hypertension == 1 |
                                 df$heart_disease == 1, "Medium Risk", "Low Risk"))

# ── Pre-compute population stats ──────────────────────────────────────────
p_stroke         <- mean(df$stroke)
p_stroke_hypert  <- mean(df$stroke[df$hypertension == 1])
p_stroke_elderly <- mean(df$stroke[df$age >= 60])
p_stroke_glucose <- mean(df$stroke[df$glucose_outlier_flag == 1])

# ── Color palette ─────────────────────────────────────────────────────────
NAV   <- as.character("#1a3a4a")
TEAL  <- as.character("#2a9d8f")
LTEAL <- as.character("#4db6ac")
NAVY2 <- as.character("#264653")
WARN  <- as.character("#e76f51")
MED   <- as.character("#e9c46a")
OK    <- as.character("#2a9d8f")

# ── Shared theme ──────────────────────────────────────────────────────────
theme_app <- function(){
  theme_minimal(base_size=13) +
    theme(
      plot.title       = element_text(face="bold", size=14, colour="#1a3a4a"),
      plot.subtitle    = element_text(size=11, colour="#555"),
      legend.position  = "bottom",
      panel.grid.minor = element_blank()
    )
}

COLOURS <- c("No Stroke"="#4db6ac", "Stroke"="#1a3a4a")

# ── UI ────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  tags$head(
    tags$style(HTML(paste0("
      body { 
        font-family: 'Segoe UI', sans-serif; 
        background-color: #f4f7f7; 
      }

      /* Header */
      .app-header {
        background: ", NAV, ";
        color: white;
        padding: 18px 20px;
        margin: -15px -15px 20px -15px;
        border-radius: 0;
      }
      .app-header h2 { margin:0; font-size:1.6em; }
      .app-header p  { margin:4px 0 0 0; font-size:0.85em; opacity:0.75; }

      /* Tabs */
      .nav-tabs > li > a {
        color: ", NAV, ";
        font-weight: bold;
      }
      .nav-tabs > li.active > a {
        color: white !important;
        background-color: ", TEAL, " !important;
        border-color: ", TEAL, " !important;
      }

      /* Section headers */
      .section-header {
        background: ", NAVY2, ";
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 15px 0 10px 0;
        font-size: 1em;
      }

      /* Cards */
      .card {
        background: white;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
      }

      /* Impact boxes */
      .impact-box {
        background: ", NAV, ";
        color: white;
        border-radius: 8px;
        padding: 18px 12px;
        margin: 6px;
        text-align: center;
        box-shadow: 0 3px 8px rgba(0,0,0,0.15);
      }
      .impact-value {
        font-size: 2.2em;
        font-weight: bold;
        color: ", LTEAL, ";
      }
      .impact-label {
        font-size: 0.82em;
        margin-top: 6px;
        opacity: 0.85;
        line-height: 1.3;
      }

      /* Metric boxes */
      .metric-box {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 6px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        border-top: 4px solid ", TEAL, ";
      }
      .metric-value {
        font-size: 1.9em;
        font-weight: bold;
        color: ", TEAL, ";
      }
      .metric-label {
        font-size: 0.82em;
        color: #555;
        margin-top: 5px;
      }

      /* Strategic cards */
      .strategic-card {
        background: white;
        border-radius: 8px;
        padding: 18px;
        margin: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
      }
      .card-high { border-top: 5px solid ", WARN,  "; }
      .card-med  { border-top: 5px solid ", MED,   "; }
      .card-low  { border-top: 5px solid ", OK,    "; }

      /* Insight box */
      .insight-box {
        background: #e8f4f3;
        border-left: 4px solid ", TEAL, ";
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 13px;
      }

      /* Patient button */
      .predict-btn {
        background-color: ", TEAL, ";
        color: white;
        border: none;
        padding: 13px 0;
        width: 100%;
        font-size: 1em;
        font-weight: bold;
        border-radius: 6px;
        cursor: pointer;
        margin-top: 12px;
        letter-spacing: 0.5px;
      }
      .predict-btn:hover {
        background-color: ", NAVY2, ";
        color: white;
      }

      /* Result card */
      .result-card {
        background: white;
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
      }

      .well { background: white; border: 1px solid #dde; }
    ")))
  ),
  
  # Header
  div(class="app-header",
      h2("Stroke Risk Prediction System"),
      p("Clinical Decision Support — Logistic Regression Model (AUC 0.827)")
  ),
  
  tabsetPanel(
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 1: EXECUTIVE DASHBOARD
    # ══════════════════════════════════════════════════════════════════
    tabPanel("Executive Dashboard",
             br(),
             
             # Impact numbers
             h4(class="section-header",
                "Why Early Stroke Detection Matters"),
             fluidRow(
               column(3, div(class="impact-box",
                             div(class="impact-value", "7.2x"),
                             div(class="impact-label",
                                 "Higher stroke risk in patients aged 60+ vs under 60")
               )),
               column(3, div(class="impact-box",
                             div(class="impact-value", "3.3x"),
                             div(class="impact-label",
                                 "Higher stroke risk in hypertensive patients")
               )),
               column(3, div(class="impact-box",
                             div(class="impact-value", "93%"),
                             div(class="impact-label",
                                 "Of real stroke cases correctly identified by this model")
               )),
               column(3, div(class="impact-box",
                             div(class="impact-value", "99.4%"),
                             div(class="impact-label",
                                 "Negative predictive value — cleared patients are safe")
               ))
             ),
             br(),
             
             # Evidence charts
             h4(class="section-header",
                "The Evidence — Stroke Risk Across Patient Groups"),
             fluidRow(
               column(4,
                      h5("Stroke Rate by Age Group",
                         style="text-align:center; font-weight:bold; color:#1a3a4a;"),
                      div(class="card",
                          withSpinner(plotOutput("exec_age_plot", height="260px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(4,
                      h5("Stroke Rate by Hypertension Status",
                         style="text-align:center; font-weight:bold; color:#1a3a4a;"),
                      div(class="card",
                          withSpinner(plotOutput("exec_hypert_plot", height="260px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(4,
                      h5("Stroke Rate by Risk Group",
                         style="text-align:center; font-weight:bold; color:#1a3a4a;"),
                      div(class="card",
                          withSpinner(plotOutput("exec_riskrate_plot", height="260px"),
                                      type=6, color=TEAL)
                      )
               )
             ),
             br(),
             
             # Strategic action plan
             h4(class="section-header", "Strategic Action Plan"),
             fluidRow(
               column(4,
                      div(class="strategic-card card-high",
                          h5("High Risk Protocol",
                             style="color:#e76f51; margin-top:0; font-size:1.05em;"),
                          p(strong("Who:"), " Patients aged 60+ with hypertension"),
                          p(strong("Stroke Rate:"), " ~13%"),
                          p(strong("Action:"),
                            " Immediate specialist referral at every consultation"),
                          p(strong("Business Case:"),
                            " Preventing one stroke avoids cost of long-term
               rehabilitation, extended inpatient stay, and
               ongoing specialist care")
                      )
               ),
               column(4,
                      div(class="strategic-card card-med",
                          h5("Medium Risk Protocol",
                             style="color:#c49a1a; margin-top:0; font-size:1.05em;"),
                          p(strong("Who:"),
                            " Aged 60+ OR hypertension OR heart disease"),
                          p(strong("Stroke Rate:"), " ~8-10%"),
                          p(strong("Action:"),
                            " Quarterly monitoring, lifestyle intervention programme"),
                          p(strong("Business Case:"),
                            " Early lifestyle intervention costs significantly less
               than acute stroke treatment and reduces readmission risk")
                      )
               ),
               column(4,
                      div(class="strategic-card card-low",
                          h5("Low Risk Protocol",
                             style="color:#2a9d8f; margin-top:0; font-size:1.05em;"),
                          p(strong("Who:"),
                            " Age under 60, no hypertension, no heart disease"),
                          p(strong("Stroke Rate:"), " ~1.8%"),
                          p(strong("Action:"),
                            " Annual standard screening, smoking history recorded"),
                          p(strong("Business Case:"),
                            " Efficient resource allocation — clinical staff time
               focused on highest risk patients")
                      )
               )
             ),
             br(),
             
             # Population overview
             h4(class="section-header", "Population Overview"),
             fluidRow(
               column(6,
                      div(class="card",
                          withSpinner(plotOutput("risk_group_plot", height="300px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(6,
                      div(class="card",
                          withSpinner(plotOutput("scatter_plot", height="300px"),
                                      type=6, color=TEAL)
                      )
               )
             )
    ),
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 2: CLINICAL DECISION SUPPORT
    # ══════════════════════════════════════════════════════════════════
    tabPanel("Clinical Decision Support",
             br(),
             div(class="insight-box",
                 "Logistic Regression was selected as the final model (AUC 0.827),
         trained on SMOTE-balanced data to address severe class imbalance
         of 4.87% stroke prevalence. At threshold 0.30, the model catches
         93 in every 100 real stroke patients."
             ),
             
             # Model metrics
             h4(class="section-header", "Model Performance Summary"),
             fluidRow(
               column(3, div(class="metric-box",
                             div(class="metric-value", "0.827"),
                             div(class="metric-label", "AUC Score")
               )),
               column(3, div(class="metric-box",
                             div(class="metric-value", "93.2%"),
                             div(class="metric-label", "Sensitivity at Threshold 0.30")
               )),
               column(3, div(class="metric-box",
                             div(class="metric-value", "58.8%"),
                             div(class="metric-label", "Specificity")
               )),
               column(3, div(class="metric-box",
                             div(class="metric-value", "99.4%"),
                             div(class="metric-label", "Negative Predictive Value")
               ))
             ),
             br(),
             
             # ROC + model comparison
             fluidRow(
               column(6,
                      h4(class="section-header", "ROC Curve — All Models"),
                      div(class="card",
                          withSpinner(plotOutput("roc_plot", height="350px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(6,
                      h4(class="section-header", "Model Comparison"),
                      div(class="card",
                          withSpinner(plotOutput("model_compare_plot", height="350px"),
                                      type=6, color=TEAL)
                      )
               )
             ),
             br(),
             
             # Feature importance + threshold
             fluidRow(
               column(6,
                      h4(class="section-header", "Feature Importance"),
                      div(class="card",
                          withSpinner(plotOutput("importance_plot", height="320px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(6,
                      h4(class="section-header", "Threshold Tuning Effect"),
                      div(class="card",
                          withSpinner(plotOutput("threshold_plot", height="320px"),
                                      type=6, color=TEAL)
                      )
               )
             ),
             br(),
             
             # Clinical interpretation
             fluidRow(
               column(12,
                      h4(class="section-header", "Clinical Interpretation"),
                      div(class="card",
                          p("Three models were trained — Logistic Regression, Decision Tree,
               and Random Forest — on SMOTE-balanced training data to correct
               for 4.87% stroke prevalence."),
                          p("Logistic Regression was selected as the winner based on highest
               AUC (0.827) and highest sensitivity (0.932) at threshold 0.30.
               Random Forest required threshold 0.10 to achieve comparable
               sensitivity, making it less stable for clinical use."),
                          p("Key predictors by odds ratio: Hypertension (10.37x),
               Heart Disease (1.78x), Active Smoking (1.46x),
               Age (11% increase per additional year)."),
                          p("The Negative Predictive Value of 99.4% means patients cleared
               by the model are almost certainly safe — only 5 stroke patients
               were missed in the test set of 74 real cases.")
                      )
               )
             )
    ),
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 3: DATA OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    tabPanel("Data Overview",
             br(),
             div(class="insight-box",
                 "5,109 patients after cleaning. 4,860 non-stroke (95.1%) and
         249 stroke (4.87%). Stroke patients are consistently older with
         higher glucose levels. Class imbalance addressed using SMOTE
         during model training."
             ),
             
             fluidRow(
               column(4,
                      div(class="card",
                          withSpinner(plotOutput("class_plot", height="280px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(8,
                      div(class="card",
                          withSpinner(plotOutput("means_plot", height="280px"),
                                      type=6, color=TEAL)
                      )
               )
             ),
             br(),
             
             fluidRow(
               column(12,
                      div(class="card",
                          fluidRow(
                            column(4,
                                   selectInput("hist_feature",
                                               "Explore feature distribution:",
                                               choices=c("age","avg_glucose_level","bmi",
                                                         "glucose_bmi","age_glucose",
                                                         "age_bmi","hypert_age"),
                                               selected="age")
                            )
                          ),
                          withSpinner(plotOutput("hist_plot", height="260px"),
                                      type=6, color=TEAL)
                      )
               )
             )
    ),
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 4: RISK FACTORS
    # ══════════════════════════════════════════════════════════════════
    tabPanel("Risk Factors",
             br(),
             div(class="insight-box",
                 "age_glucose is the strongest engineered feature (importance 78.21),
         outperforming raw age and raw glucose individually. Stroke cases
         cluster visibly in the high-age, high-glucose region."
             ),
             
             fluidRow(
               column(7,
                      div(class="card",
                          fluidRow(
                            column(6,
                                   selectInput("x_axis", "X axis:",
                                               choices=c("age","avg_glucose_level","bmi",
                                                         "age_glucose","glucose_bmi"),
                                               selected="age")
                            ),
                            column(6,
                                   selectInput("y_axis", "Y axis:",
                                               choices=c("avg_glucose_level","age","bmi",
                                                         "age_glucose","glucose_bmi"),
                                               selected="avg_glucose_level")
                            )
                          ),
                          withSpinner(plotOutput("scatter_rf", height="320px"),
                                      type=6, color=TEAL)
                      )
               ),
               column(5,
                      div(class="card",
                          withSpinner(plotOutput("importance_plot2", height="380px"),
                                      type=6, color=TEAL)
                      )
               )
             ),
             br(),
             
             fluidRow(
               column(12,
                      div(class="card",
                          withSpinner(plotOutput("age_density_plot", height="280px"),
                                      type=6, color=TEAL)
                      )
               )
             )
    ),
    
    # ══════════════════════════════════════════════════════════════════
    # TAB 5: PATIENT RISK ASSESSMENT
    # ══════════════════════════════════════════════════════════════════
    tabPanel("My Risk Assessment",
             br(),
             fluidRow(
               column(4,
                      wellPanel(
                        h4("Enter Your Details",
                           style="color:#1a3a4a; margin-top:0;"),
                        sliderInput("age_in", "Age (years)",
                                    min=1, max=82, value=45),
                        sliderInput("glucose_in", "Average Glucose (mg/dL)",
                                    min=55, max=272, value=100),
                        sliderInput("bmi_in", "BMI (kg/m²)",
                                    min=10, max=98, value=28),
                        selectInput("hypertension_in",
                                    "Do you have hypertension?",
                                    choices=c("No"=0, "Yes"=1)),
                        selectInput("heart_disease_in",
                                    "Do you have heart disease?",
                                    choices=c("No"=0, "Yes"=1)),
                        selectInput("gender_in", "Gender",
                                    choices=c("Female"=0, "Male"=1)),
                        selectInput("ever_married_in", "Ever Married?",
                                    choices=c("No"=0, "Yes"=1)),
                        selectInput("residence_in", "Where do you live?",
                                    choices=c("Rural"=0, "Urban"=1)),
                        selectInput("work_type_in", "Work Type",
                                    choices=c("Private","Self-employed",
                                              "Govt_job","children",
                                              "Never_worked")),
                        selectInput("smoking_in", "Smoking Status",
                                    choices=c("never smoked","formerly smoked",
                                              "smokes","Unknown")),
                        tags$button("Calculate My Risk",
                                    id="predict",
                                    class="predict-btn action-button",
                                    type="button")
                      )
               ),
               column(8,
                      uiOutput("patient_result"),
                      br(),
                      h4(class="section-header", "What Does This Mean?"),
                      div(class="card", uiOutput("patient_explanation")),
                      br(),
                      h4(class="section-header", "Your Key Risk Factors"),
                      div(class="card", uiOutput("risk_factors")),
                      br(),
                      p(style="color:#888; font-size:0.8em;",
                        "Note: This tool is for educational purposes only and is
             not a clinical diagnostic tool. Always consult a medical
             professional.")
               )
             )
    )
  )
)

# ── SERVER ────────────────────────────────────────────────────────────────
server <- function(input, output){
  
  # ── Prediction ──────────────────────────────────────────────────────────
  prediction <- eventReactive(input$predict, {
    new_patient <- data.frame(
      age               = as.numeric(input$age_in),
      avg_glucose_level = as.numeric(input$glucose_in),
      bmi               = as.numeric(input$bmi_in),
      hypertension      = as.numeric(input$hypertension_in),
      heart_disease     = as.numeric(input$heart_disease_in),
      gender            = as.numeric(input$gender_in),
      ever_married      = as.numeric(input$ever_married_in),
      Residence_type    = as.numeric(input$residence_in),
      work_typechildren    = as.integer(input$work_type_in=="children"),
      work_typeGovt_job    = as.integer(input$work_type_in=="Govt_job"),
      work_typeNever_worked= as.integer(input$work_type_in=="Never_worked"),
      work_typePrivate     = as.integer(input$work_type_in=="Private"),
      work_typeSelf.employed = as.integer(input$work_type_in=="Self-employed"),
      smoking_statusformerly.smoked =
        as.integer(input$smoking_in=="formerly smoked"),
      smoking_statusnever.smoked =
        as.integer(input$smoking_in=="never smoked"),
      smoking_statussmokes =
        as.integer(input$smoking_in=="smokes"),
      smoking_statusUnknown =
        as.integer(input$smoking_in=="Unknown"),
      glucose_outlier_flag =
        as.integer(as.numeric(input$glucose_in) > 145),
      glucose_bmi =
        as.numeric(input$glucose_in) * as.numeric(input$bmi_in),
      age_glucose =
        as.numeric(input$age_in) * as.numeric(input$glucose_in),
      age_bmi     =
        as.numeric(input$age_in) * as.numeric(input$bmi_in),
      hypert_age  =
        as.numeric(input$hypertension_in) * as.numeric(input$age_in)
    )
    prob    <- predict(model, new_patient, type="prob")[,"Stroke"]
    outcome <- ifelse(prob >= 0.30, "HIGH RISK", "LOW RISK")
    list(prob=prob, outcome=outcome)
  })
  
  # ── Patient outputs ──────────────────────────────────────────────────────
  output$patient_result <- renderUI({
    req(prediction())
    p   <- prediction()
    col <- ifelse(p$outcome=="HIGH RISK", WARN, TEAL)
    div(class="result-card",
        style=paste0("border-left:6px solid ", col, ";"),
        h3(p$outcome, style=paste0("color:", col, "; margin:0;")),
        h4(paste0("Stroke Probability: ", round(p$prob*100, 1), "%"),
           style="margin:10px 0 0 0; color:#333;")
    )
  })
  
  output$patient_explanation <- renderUI({
    req(prediction())
    p <- prediction()
    if(p$outcome=="HIGH RISK"){
      div(
        p("Your results suggest you may be at elevated risk of stroke.
           This does not mean you will have a stroke — it means you
           should speak to your doctor as soon as possible for a full
           clinical assessment."),
        p("Early detection and lifestyle changes can significantly
           reduce your risk.")
      )
    } else {
      div(
        p("Your results suggest you are currently at lower risk of
           stroke. Continue maintaining a healthy lifestyle and attend
           your regular annual check-ups."),
        p("If your health circumstances change, please reassess.")
      )
    }
  })
  
  output$risk_factors <- renderUI({
    req(input$predict)
    factors <- c()
    if(as.numeric(input$age_in) >= 60)
      factors <- c(factors, "Age 60+ — 7.2x higher stroke risk")
    if(as.numeric(input$hypertension_in)==1)
      factors <- c(factors, "Hypertension — 3.3x higher stroke risk")
    if(as.numeric(input$heart_disease_in)==1)
      factors <- c(factors, "Heart Disease — 78% higher stroke risk")
    if(input$smoking_in=="smokes")
      factors <- c(factors, "Active Smoker — 46% higher stroke risk")
    if(as.numeric(input$glucose_in) > 145)
      factors <- c(factors, "High Glucose — 3.6x higher stroke risk")
    if(length(factors)==0)
      factors <- c("No major risk factors identified")
    div(tags$ul(lapply(factors, tags$li)))
  })
  
  # ── Executive plots ──────────────────────────────────────────────────────
  output$exec_age_plot <- renderPlot({
    age_data <- data.frame(
      Group = c("Under 40","40-59","60+"),
      Rate  = c(
        round(mean(df$stroke[df$age < 40])*100, 1),
        round(mean(df$stroke[df$age>=40 & df$age<60])*100, 1),
        round(mean(df$stroke[df$age >= 60])*100, 1)
      )
    )
    age_data$Group <- factor(age_data$Group,
                             levels=c("Under 40","40-59","60+"))
    ggplot(age_data, aes(x=Group, y=Rate, fill=Group)) +
      geom_bar(stat="identity", width=0.55) +
      geom_text(aes(label=paste0(Rate,"%")),
                vjust=-0.5, fontface="bold", size=4.5) +
      scale_fill_manual(values=c("Under 40"=LTEAL,
                                 "40-59"=TEAL,
                                 "60+"=NAV)) +
      labs(x="Age Group", y="Stroke Rate (%)") +
      theme_app() +
      theme(legend.position="none") +
      ylim(0, max(age_data$Rate)*1.25)
  })
  
  output$exec_hypert_plot <- renderPlot({
    hyp_data <- data.frame(
      Group = c("No Hypertension","Hypertension"),
      Rate  = c(
        round(mean(df$stroke[df$hypertension==0])*100, 1),
        round(mean(df$stroke[df$hypertension==1])*100, 1)
      )
    )
    ggplot(hyp_data, aes(x=Group, y=Rate, fill=Group)) +
      geom_bar(stat="identity", width=0.45) +
      geom_text(aes(label=paste0(Rate,"%")),
                vjust=-0.5, fontface="bold", size=4.5) +
      scale_fill_manual(values=c("No Hypertension"=LTEAL,
                                 "Hypertension"=NAV)) +
      labs(x="", y="Stroke Rate (%)") +
      theme_app() +
      theme(legend.position="none") +
      ylim(0, max(hyp_data$Rate)*1.25)
  })
  
  output$exec_riskrate_plot <- renderPlot({
    rg_rate <- df %>%
      group_by(risk_group) %>%
      summarise(stroke_rate=round(mean(stroke)*100, 1),
                .groups="drop")
    ggplot(rg_rate,
           aes(x=reorder(risk_group,-stroke_rate),
               y=stroke_rate, fill=risk_group)) +
      geom_bar(stat="identity", width=0.45) +
      geom_text(aes(label=paste0(stroke_rate,"%")),
                vjust=-0.5, fontface="bold", size=4.5) +
      scale_fill_manual(values=c("High Risk"=NAV,
                                 "Medium Risk"=TEAL,
                                 "Low Risk"=LTEAL)) +
      labs(x="Risk Group", y="Stroke Rate (%)") +
      theme_app() +
      theme(legend.position="none") +
      ylim(0, max(rg_rate$stroke_rate)*1.25)
  })
  
  output$risk_group_plot <- renderPlot({
    rg <- as.data.frame(table(df$risk_group))
    names(rg) <- c("Group","Count")
    ggplot(rg, aes(x=reorder(Group,-Count), y=Count, fill=Group)) +
      geom_bar(stat="identity") +
      geom_text(aes(label=Count), vjust=-0.5, fontface="bold") +
      scale_fill_manual(values=c("High Risk"=NAV,
                                 "Medium Risk"=TEAL,
                                 "Low Risk"=LTEAL)) +
      labs(title="Patient Distribution by Risk Group",
           x="Risk Group", y="Number of Patients") +
      theme_app() +
      theme(legend.position="none")
  })
  
  output$scatter_plot <- renderPlot({
    ggplot(df, aes(x=age, y=avg_glucose_level, colour=stroke_f)) +
      geom_point(alpha=0.35, size=1.3) +
      scale_colour_manual(values=COLOURS) +
      labs(title="Age vs Glucose by Stroke Outcome",
           x="Age", y="Glucose (mg/dL)", colour="Outcome") +
      theme_app()
  })
  
  # ── Clinical plots ───────────────────────────────────────────────────────
  output$roc_plot <- renderPlot({
    roc_lr <- roc(roc_data$actual, roc_data$prob_lr,
                  levels=c("No_Stroke","Stroke"), quiet=TRUE)
    roc_dt <- roc(roc_data$actual, roc_data$prob_dt,
                  levels=c("No_Stroke","Stroke"), quiet=TRUE)
    roc_rf <- roc(roc_data$actual, roc_data$prob_rf,
                  levels=c("No_Stroke","Stroke"), quiet=TRUE)
    
    roc_df <- rbind(
      data.frame(FPR=1-roc_lr$specificities,
                 TPR=roc_lr$sensitivities,
                 Model=paste0("Logistic Regression (AUC=",
                              round(auc(roc_lr),3),")")),
      data.frame(FPR=1-roc_dt$specificities,
                 TPR=roc_dt$sensitivities,
                 Model=paste0("Decision Tree (AUC=",
                              round(auc(roc_dt),3),")")),
      data.frame(FPR=1-roc_rf$specificities,
                 TPR=roc_rf$sensitivities,
                 Model=paste0("Random Forest (AUC=",
                              round(auc(roc_rf),3),")"))
    )
    
    ggplot(roc_df, aes(x=FPR, y=TPR, colour=Model)) +
      geom_line(linewidth=1.2) +
      geom_abline(slope=1, intercept=0,
                  linetype="dashed", colour="grey60") +
      scale_colour_manual(values=setNames(
        c(NAV, TEAL, LTEAL),
        unique(roc_df$Model)
      )) +
      labs(title="ROC Curves — Model Comparison",
           subtitle="Higher and further left = better discrimination",
           x="False Positive Rate (1 - Specificity)",
           y="True Positive Rate (Sensitivity)",
           colour="") +
      theme_app()
  })
  
  output$model_compare_plot <- renderPlot({
    metrics <- data.frame(
      Model  = rep(c("Logistic Regression",
                     "Decision Tree",
                     "Random Forest"), 4),
      Metric = rep(c("AUC","Sensitivity",
                     "Specificity","Accuracy"), each=3),
      Value  = c(
        0.827, 0.753, 0.806,   # AUC
        0.932, 0.878, 0.905,   # Sensitivity (LR@0.30, DT@0.30, RF@0.10)
        0.588, 0.627, 0.587,   # Specificity
        0.604, 0.639, 0.603    # Accuracy
      )
    )
    metrics$Model <- factor(metrics$Model,
                            levels=c("Logistic Regression",
                                     "Decision Tree",
                                     "Random Forest"))
    ggplot(metrics, aes(x=Model, y=Value, fill=Model)) +
      geom_bar(stat="identity", width=0.6) +
      geom_text(aes(label=round(Value,3)),
                vjust=-0.4, size=3.2, fontface="bold") +
      facet_wrap(~Metric, nrow=1) +
      scale_fill_manual(values=c(
        "Logistic Regression" = NAV,
        "Decision Tree"       = TEAL,
        "Random Forest"       = LTEAL
      )) +
      scale_y_continuous(expand=expansion(mult=c(0,0.2)),
                         limits=c(0,1)) +
      labs(title="Model Comparison Across All Metrics",
           subtitle="Logistic Regression wins on AUC and Sensitivity",
           x="", y="Score") +
      theme_app() +
      theme(legend.position="none",
            axis.text.x=element_text(angle=25, hjust=1, size=9),
            strip.text=element_text(face="bold", size=11))
  })
  
  output$importance_plot <- renderPlot({
    imp_data <- data.frame(
      Feature=c("age","age_glucose","age_bmi",
                "smoking_statusformerly.smoked","ever_married",
                "smoking_statusnever.smoked","heart_disease",
                "hypert_age","work_typeSelf.employed",
                "Residence_type","work_typePrivate",
                "hypertension","gender","avg_glucose_level",
                "glucose_bmi"),
      Score=c(100,78.21,58.99,41.45,40.37,
              36.91,33.68,32.44,30.27,
              30.13,29.64,28.81,25.97,24.76,23.41)
    )
    ggplot(imp_data,
           aes(x=reorder(Feature,Score), y=Score,
               fill=Score > 50)) +
      geom_bar(stat="identity") +
      geom_text(aes(label=round(Score,1)),
                hjust=-0.1, size=3.2) +
      coord_flip() +
      scale_fill_manual(values=c("TRUE"=NAV, "FALSE"=TEAL)) +
      scale_y_continuous(expand=expansion(mult=c(0,0.18))) +
      labs(title="Random Forest Feature Importance",
           subtitle="Engineered features rank above raw clinical variables",
           x="", y="Importance Score") +
      theme_app() +
      theme(legend.position="none")
  })
  
  output$threshold_plot <- renderPlot({
    thresh_data <- data.frame(
      Metric    = rep(c("Sensitivity","Specificity","Accuracy"), 2),
      Threshold = rep(c("Default (0.50)","Tuned (0.30)"), each=3),
      Value     = c(0.486, 0.932, 0.872,
                    0.932, 0.588, 0.604)
    )
    thresh_data$Threshold <- factor(thresh_data$Threshold,
                                    levels=c("Default (0.50)",
                                             "Tuned (0.30)"))
    ggplot(thresh_data,
           aes(x=Metric, y=Value, fill=Threshold)) +
      geom_bar(stat="identity", position="dodge", width=0.6) +
      geom_text(aes(label=paste0(round(Value*100,1),"%")),
                position=position_dodge(width=0.6),
                vjust=-0.4, size=3.5, fontface="bold") +
      scale_fill_manual(values=c("Default (0.50)"=LTEAL,
                                 "Tuned (0.30)"=NAV)) +
      scale_y_continuous(limits=c(0,1.15),
                         expand=expansion(mult=c(0,0.1))) +
      labs(title="Effect of Threshold Tuning",
           subtitle="Lowering to 0.30 catches 93% of real stroke patients",
           x="", y="Score", fill="Threshold") +
      theme_app()
  })
  
  # ── Data overview plots ──────────────────────────────────────────────────
  output$class_plot <- renderPlot({
    counts <- as.data.frame(table(df$stroke_f))
    names(counts) <- c("Outcome","Count")
    counts$Pct <- round(counts$Count/sum(counts$Count)*100,1)
    ggplot(counts, aes(x=Outcome, y=Count, fill=Outcome)) +
      geom_bar(stat="identity", width=0.5) +
      geom_text(aes(label=paste0(Count,"\n(",Pct,"%)")),
                vjust=-0.3, fontface="bold", size=4) +
      scale_fill_manual(values=COLOURS) +
      scale_y_continuous(expand=expansion(mult=c(0,0.2))) +
      labs(title="Class Distribution", x="", y="Count") +
      theme_app() +
      theme(legend.position="none")
  })
  
  output$means_plot <- renderPlot({
    means <- df %>%
      group_by(stroke_f) %>%
      summarise(Age=mean(age),
                Glucose=mean(avg_glucose_level),
                BMI=mean(bmi),
                .groups="drop") %>%
      melt(id.vars="stroke_f",
           variable.name="Feature",
           value.name="Mean")
    ggplot(means, aes(x=Feature, y=Mean, fill=stroke_f)) +
      geom_bar(stat="identity", position="dodge", width=0.6) +
      geom_text(aes(label=round(Mean,1)),
                position=position_dodge(width=0.6),
                vjust=-0.4, size=3.5, fontface="bold") +
      scale_fill_manual(values=COLOURS) +
      scale_y_continuous(expand=expansion(mult=c(0,0.15))) +
      labs(title="Mean Values by Outcome",
           subtitle="Stroke patients are older with higher glucose",
           x="", y="Mean Value", fill="Outcome") +
      theme_app()
  })
  
  output$hist_plot <- renderPlot({
    feat <- input$hist_feature
    ggplot(df, aes_string(x=feat, fill="stroke_f")) +
      geom_histogram(bins=28, alpha=0.65, position="identity") +
      scale_fill_manual(values=COLOURS) +
      labs(title=paste("Distribution of", feat, "by Outcome"),
           subtitle="Overlap shows where prediction is hardest",
           x=feat, y="Count", fill="Outcome") +
      theme_app()
  })
  
  # ── Risk factors plots ───────────────────────────────────────────────────
  output$scatter_rf <- renderPlot({
    ggplot(df, aes_string(x=input$x_axis,
                          y=input$y_axis,
                          colour="stroke_f")) +
      geom_point(alpha=0.35, size=1.5) +
      geom_smooth(method="lm", se=TRUE, linewidth=1) +
      scale_colour_manual(values=COLOURS) +
      labs(title=paste(input$x_axis, "vs", input$y_axis),
           subtitle="Stroke cases cluster in high-risk region",
           colour="Outcome") +
      theme_app()
  })
  
  output$importance_plot2 <- renderPlot({
    imp_data <- data.frame(
      Feature=c("age","age_glucose","age_bmi",
                "smoking_statusformerly.smoked","ever_married",
                "smoking_statusnever.smoked","heart_disease",
                "hypert_age","work_typeSelf.employed",
                "Residence_type","work_typePrivate",
                "hypertension","gender","avg_glucose_level",
                "glucose_bmi"),
      Score=c(100,78.21,58.99,41.45,40.37,
              36.91,33.68,32.44,30.27,
              30.13,29.64,28.81,25.97,24.76,23.41)
    )
    ggplot(imp_data,
           aes(x=reorder(Feature,Score), y=Score,
               fill=Score > 50)) +
      geom_bar(stat="identity") +
      geom_text(aes(label=round(Score,1)),
                hjust=-0.1, size=3.2) +
      coord_flip() +
      scale_fill_manual(values=c("TRUE"=NAV,"FALSE"=TEAL)) +
      scale_y_continuous(expand=expansion(mult=c(0,0.18))) +
      labs(title="Feature Importance",
           subtitle="Top 15 predictors",
           x="", y="Importance Score") +
      theme_app() +
      theme(legend.position="none")
  })
  
  output$age_density_plot <- renderPlot({
    ggplot(df, aes(x=age, fill=stroke_f)) +
      geom_density(alpha=0.6) +
      scale_fill_manual(values=COLOURS) +
      geom_vline(xintercept=60, linetype="dashed",
                 colour="#333", linewidth=0.8) +
      annotate("text", x=62, y=0.035,
               label="Risk accelerates\nafter age 60",
               hjust=0, size=3.5, colour="#333") +
      labs(title="Age Distribution by Outcome",
           subtitle="Stroke patients skew significantly older",
           x="Age", y="Density", fill="Outcome") +
      theme_app()
  })
}

shinyApp(ui, server)
