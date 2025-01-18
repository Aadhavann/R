library(tidyverse)
library(text2vec)
library(stringr)
library(caret)
library(glmnet)

extract_features <- function(student_answer, model_answer) {
  # Clean and tokenize text
  clean_text <- function(text) {
    text %>%
      tolower() %>%
      str_replace_all("[[:punct:]]", " ") %>%
      str_replace_all("[[:digit:]]", " ") %>%
      str_replace_all("\\s+", " ") %>%
      str_trim()
  }
  
  student_clean <- clean_text(student_answer)
  model_clean <- clean_text(model_answer)
  
  # Create document-term matrices
  it_train <- itoken(c(student_clean, model_clean),
                    preprocessor = identity,
                    tokenizer = word_tokenizer)
  
  vocab <- create_vocabulary(it_train)
  vectorizer <- vocab_vectorizer(vocab)
  dtm <- as.matrix(create_dtm(it_train, vectorizer))  # Convert to dense matrix
  
  # Calculate cosine similarity
  if (ncol(dtm) > 0) {
    cos_sim <- cosine(dtm[1,], dtm[2,])
  } else {
    cos_sim <- 0  # Default if no common terms
  }
  
  # Calculate length-based features
  length_ratio <- nchar(student_clean) / nchar(model_clean)
  word_count_ratio <- length(strsplit(student_clean, "\\s+")[[1]]) / 
                      length(strsplit(model_clean, "\\s+")[[1]])
  
  # Calculate word overlap
  student_words <- unique(strsplit(student_clean, "\\s+")[[1]])
  model_words <- unique(strsplit(model_clean, "\\s+")[[1]])
  overlap <- length(intersect(student_words, model_words)) / 
            length(union(student_words, model_words))
  
  # Return features as a named vector
  c(
    cosine_similarity = cos_sim,
    length_ratio = length_ratio,
    word_count_ratio = word_count_ratio,
    word_overlap = overlap
  )
}

cosine <- function(x, y) {
  sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)))
}

create_training_data <- function(answers_df) {
  features_list <- mapply(
    extract_features,
    answers_df$student_answer,
    answers_df$model_answer,
    SIMPLIFY = FALSE
  )
  
  features_df <- do.call(rbind, features_list) %>%
    as.data.frame()
  
  cbind(answers_df, features_df)
}

train_ensemble <- function(train_data) {
  # Train base models
  model_rf <- train(
    x = train_data %>% select(cosine_similarity:word_overlap),
    y = train_data$score,
    method = "rf",
    trControl = trainControl(method = "cv", number = 5)
  )
  
  model_glm <- train(
    x = train_data %>% select(cosine_similarity:word_overlap),
    y = train_data$score,
    method = "glmnet",
    trControl = trainControl(method = "cv", number = 5)
  )
  
  # Make predictions for meta-features
  pred_rf <- predict(model_rf, train_data)
  pred_glm <- predict(model_glm, train_data)
  
  # Train meta-model
  meta_features <- data.frame(
    rf_pred = pred_rf,
    glm_pred = pred_glm
  )
  
  meta_model <- train(
    x = meta_features,
    y = train_data$score,
    method = "glmnet",
    trControl = trainControl(method = "cv", number = 5)
  )
  
  list(
    rf = model_rf,
    glm = model_glm,
    meta = meta_model
  )
}

predict_ensemble <- function(ensemble, new_data) {
  # Get base model predictions
  pred_rf <- predict(ensemble$rf, new_data)
  pred_glm <- predict(ensemble$glm, new_data)
  
  # Create meta-features
  meta_features <- data.frame(
    rf_pred = pred_rf,
    glm_pred = pred_glm
  )
  
  # Final prediction
  predict(ensemble$meta, meta_features)
}

sample_data <- data.frame(
  student_answer = c(
    "The process of photosynthesis converts light energy into chemical energy",
    "Photosynthesis is when plants make food using sunlight",
    "Plants use CO2 and water to create glucose through photosynthesis"
  ),
  model_answer = rep("Photosynthesis is the process by which plants convert light energy into chemical energy to produce glucose from carbon dioxide and water", 3),
  score = c(1.0, 0.7, 0.8)
)

training_data <- create_training_data(sample_data)

# Train ensemble
ensemble_model <- train_ensemble(training_data)

# Make predictions
predictions <- predict_ensemble(ensemble_model, training_data)

# Calculate RMSE
rmse <- sqrt(mean((predictions - training_data$score)^2))
print(paste("RMSE:", round(rmse, 3)))

new_answer <- data.frame(
  student_answer = "Photosynthesis uses sunlight to make energy in plants",
  model_answer = "Photosynthesis is the process by which plants convert light energy into chemical energy to produce glucose from carbon dioxide and water"
)

# Predict score for new answer
new_features <- create_training_data(new_answer)
predicted_score <- predict_ensemble(ensemble_model, new_features)
print(paste("Predicted score:", round(predicted_score, 2)))