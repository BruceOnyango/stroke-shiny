library(testthat)
library(httr)

# Ensure paths resolve from project root
if(basename(getwd()) == "tests") setwd("..")

BASE_URL <- "https://bruceonyango.link/stroke"

test_that("App is reachable", {
  response <- GET(BASE_URL)
  expect_equal(status_code(response), 200)
})

test_that("App returns HTML content", {
  response <- GET(BASE_URL)
  content_type <- headers(response)$`content-type`
  expect_true(grepl("text/html", content_type))
})

test_that("App response time is under 10 seconds", {
  start   <- Sys.time()
  GET(BASE_URL)
  elapsed <- as.numeric(Sys.time() - start)
  expect_lt(elapsed, 10)
})

test_that("App contains expected content", {
  response  <- GET(BASE_URL)
  body_text <- content(response, "text")
  expect_true(grepl("Stroke Risk Prediction", body_text))
  expect_true(grepl("Executive Dashboard", body_text))
  expect_true(grepl("Clinical Decision Support", body_text))
  expect_true(grepl("My Risk Assessment", body_text))
})

cat("\n========================================\n")
cat("PRODUCTION SMOKE TESTS\n")
cat("========================================\n")
cat("All production tests completed.\n")