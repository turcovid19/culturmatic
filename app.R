library(shiny)
library(readxl)
library(broom)
library(dplyr)
library(DT)
library(caret)
library(pROC)

ui <- fluidPage(
  titlePanel("Logistic Regression App"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload Excel File", accept = ".xlsx"),
      uiOutput("var_select_y"),
      uiOutput("var_select_x"),
      uiOutput("var_select_types"),
      numericInput("kfold", "K-Fold Number:", value = 5, min = 2)
    ),
    mainPanel(
      h4("Model Results (K-Fold Average)"),
      dataTableOutput("model_table"),
      h4("Classification Performance (Average)"),
      tableOutput("performance_table")
    )
  )
)

server <- function(input, output, session) {
  data <- reactive({
    req(input$file)
    df <- read_excel(input$file$datapath)
    names(df) <- make.names(names(df))
    return(df)
  })
  
  output$var_select_y <- renderUI({
    req(data())
    selectInput("dependent", "Dependent Variable (must be binary):", 
                choices = names(data()), selected = NULL)
  })
  
  output$var_select_x <- renderUI({
    req(data())
    selectizeInput("independents", "Independent Variable(s):", 
                   choices = names(data()), multiple = TRUE)
  })
  
  output$var_select_types <- renderUI({
    req(input$independents)
    lapply(input$independents, function(var) {
      selectInput(paste0("type_", var), paste("Variable type for", var),
                  choices = c("Categorical" = "factor", "Continuous" = "numeric"), selected = "factor")
    })
  })
  
  output$model_table <- renderDataTable({
    req(input$dependent, input$independents, input$kfold)
    df <- data() %>% 
      select(all_of(c(input$dependent, input$independents))) %>% 
      na.omit()
    
    for (var in input$independents) {
      var_type <- input[[paste0("type_", var)]]
      if (!is.null(var_type) && var_type == "factor") {
        df[[var]] <- as.factor(df[[var]])
      } else {
        df[[var]] <- as.numeric(df[[var]])
      }
    }
    
    df[[input$dependent]] <- factor(df[[input$dependent]])
    formula <- as.formula(paste(input$dependent, "~", paste(input$independents, collapse = "+")))
    
    set.seed(123)
    folds <- createFolds(df[[input$dependent]], k = input$kfold, list = TRUE, returnTrain = FALSE)
    
    results <- list()
    perf_metrics <- data.frame(Accuracy = numeric(), Sensitivity = numeric(), Specificity = numeric(), 
                               PPV = numeric(), NPV = numeric(), AUC = numeric())
    
    for (i in seq_along(folds)) {
      test_idx <- folds[[i]]
      train_data <- df[-test_idx, ]
      test_data <- df[test_idx, ]
      
      model <- glm(formula, data = train_data, family = binomial)
      probs <- predict(model, newdata = test_data, type = "response")
      preds <- ifelse(probs > 0.5, 1, 0)
      actuals <- as.numeric(as.character(test_data[[input$dependent]]))
      
      cm <- confusionMatrix(factor(preds), factor(actuals), positive = "1")
      roc_obj <- roc(actuals, probs)
      
      perf_metrics[i, ] <- c(
        cm$overall["Accuracy"],
        cm$byClass["Sensitivity"],
        cm$byClass["Specificity"],
        cm$byClass["Pos Pred Value"],
        cm$byClass["Neg Pred Value"],
        auc(roc_obj)
      )
      
      fold_result <- tidy(model, conf.int = TRUE) %>%
        mutate(ExpB = exp(estimate)) %>%
        select(term, estimate, ExpB, conf.low, conf.high, p.value)
      
      results[[i]] <- fold_result
    }
    
    avg_results <- bind_rows(results) %>%
      group_by(term) %>%
      summarise(
        B = mean(estimate, na.rm = TRUE),
        ExpB = mean(ExpB, na.rm = TRUE),
        CI_Lower = mean(conf.low, na.rm = TRUE),
        CI_Upper = mean(conf.high, na.rm = TRUE),
        p_value = mean(p.value, na.rm = TRUE)
      ) %>%
      rename(Variable = term) %>%
      mutate(
        B = round(B, 3),
        ExpB = round(ExpB, 3),
        CI_Lower = round(CI_Lower, 3),
        CI_Upper = round(CI_Upper, 3),
        p_value = round(p_value, 3)
      )
    
    output$performance_table <- renderTable({
      perf_summary <- colMeans(perf_metrics, na.rm = TRUE)
      round(as.data.frame(t(perf_summary)), 3)
    }, rownames = TRUE)
    
    datatable(avg_results, options = list(pageLength = 10), rownames = FALSE)
  })
}

shinyApp(ui, server)
