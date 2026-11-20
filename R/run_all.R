#!/usr/bin/env Rscript

args_full <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_full, value = TRUE)
if (length(file_arg) == 0) stop("Run this file with Rscript R/run_all.R")
script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
root <- normalizePath(file.path(dirname(script_path), ".."))
data_path <- file.path(root, "data", "student_performance.csv")
out_dir <- file.path(root, "results", "R")
fig_dir <- file.path(out_dir, "figures")
if (!file.exists(data_path)) {
  stop("Dataset not found: ", data_path, ". Run python/generate_data.py first.")
}
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
data <- read.csv(data_path, stringsAsFactors = FALSE)
required <- c(
  "date", "group", "gender", "study_hours", "attendance_pct",
  "score", "passed", "score_complete"
)
missing_columns <- setdiff(required, names(data))
if (length(missing_columns)) {
  stop("Dataset is missing required columns: ", paste(missing_columns, collapse = ", "))
}

topic_titles <- c(
  "Measures of central tendency and dispersion",
  "Probability and conditional probability",
  "Probability distributions",
  "Sampling distributions and Central Limit Theorem",
  "Confidence intervals",
  "Hypothesis testing",
  "Correlation and covariance",
  "Linear regression and least squares",
  "Matrix algebra in statistics",
  "Eigenvalues, eigenvectors and PCA",
  "Numerical optimization",
  "Random numbers and Monte Carlo methods",
  "Time-series analysis",
  "Statistical data visualization",
  "Data transformation and normalization",
  "ANOVA and experimental design",
  "Chi-square and categorical data analysis",
  "Logistic regression",
  "Non-parametric statistical methods",
  "Bootstrap and resampling methods",
  "Multivariate statistical analysis",
  "Missing-data analysis and imputation",
  "Statistical reporting and reproducible analysis"
)
set.seed(4131)

save_plot <- function(number, draw, width = 7, height = 5) {
  grDevices::png(
    file.path(fig_dir, sprintf("topic_%02d.png", number)),
    width = width, height = height, units = "in", res = 150
  )
  draw()
  grDevices::dev.off()
}

topic1 <- function() {
  x <- data$score_complete
  q <- quantile(x, c(0.25, 0.75))
  save_plot(1, function() {
    boxplot(x, horizontal = TRUE, xlab = "Score",
            main = "Complete assessment-score distribution")
  )
  paste0(
    "n=", length(x), "; mean=", round(mean(x), 3),
    "; median=", round(median(x), 3),
    "; mode=", names(which.max(table(x))),
    "; sample variance=", round(var(x), 3),
    "; SD=", round(sd(x), 3), "; Q1=", round(q[[1]], 2),
    "; Q3=", round(q[[2]], 2), "; IQR=", round(IQR(x), 2),
    ". Mean/median and IQR describe location and spread."
  )
}

topic2 <- function() {
  d <- data[!is.na(data$passed), ]
  p <- mean(d$passed)
  conditional <- mean(d$passed[d$group == "B"])
  simulated <- rbinom(10000, size = 1, prob = p)
  paste0(
    "P(pass)=", round(p, 4), "; P(pass | group B)=",
    round(conditional, 4), "; 10,000-trial simulated proportion=",
    round(mean(simulated), 4),
    ". The simulated rate varies around the observed rate."
  )
}

topic3 <- function() {
  x <- data$score[!is.na(data$score)]
  mu <- mean(x)
  sigma <- sd(x)
  probability <- pnorm(60, mean = mu, sd = sigma)
  save_plot(3, function() {
    hist(x, probability = TRUE, breaks = 18, col = "grey80",
         main = "Normal fit to observed scores", xlab = "Score")
    curve(dnorm(x, mean = mu, sd = sigma), add = TRUE, col = "red", lwd = 2)
  )
  paste0(
    "Fitted normal mu=", round(mu, 3), ", sigma=", round(sigma, 3),
    "; fitted P(score <= 60)=", round(probability, 4),
    ". The normal model is an approximation to bounded scores."
  )
}

topic4 <- function() {
  x <- data$score_complete
  means <- replicate(5000, mean(sample(x, 30, replace = TRUE)))
  expected_se <- sd(x) / sqrt(30)
  save_plot(4, function() {
    hist(means, breaks = 35, col = "grey80",
         main = "Sampling distribution of means (n=30)",
         xlab = "Sample mean")
    abline(v = mean(x), col = "red", lty = 2)
  )
  paste0(
    "5,000 resamples of size 30: mean of sample means=",
    round(mean(means), 3), "; simulated SE=", round(sd(means), 3),
    "; theoretical SE=", round(expected_se, 3),
    ". Sample means are more concentrated and approximately normal."
  )
}

topic5 <- function() {
  x <- data$score_complete
  interval <- t.test(x, conf.level = 0.95)$conf.int
  paste0(
    "Mean=", round(mean(x), 3), "; 95% t interval=(",
    round(interval[[1]], 3), ", ", round(interval[[2]], 3),
    "); under the sampling assumptions, the method has 95% long-run coverage."
  )
}

topic6 <- function() {
  x <- data$score_complete
  test <- t.test(x, mu = 60)
  decision <- if (test$p.value < 0.05) "reject H0" else "fail to reject H0"
  paste0(
    "H0: mean score=60; t(", length(x) - 1, ")=",
    round(unname(test$statistic), 4), "; two-sided p=",
    format.pval(test$p.value, digits = 5), "; alpha=0.05: ", decision,
    ". A p-value is not the probability that H0 is true."
  )
}

topic7 <- function() {
  d <- data[complete.cases(data[, c("study_hours", "score_complete")]), ]
  cov_value <- cov(d$study_hours, d$score_complete)
  correlation <- cor(d$study_hours, d$score_complete)
  save_plot(7, function() {
    plot(d$study_hours, d$score_complete, pch = 19, col = "#33669988",
         xlab = "Study hours", ylab = "Score",
         main = "Study hours and assessment score")
  )
  paste0(
    "Complete pairs n=", nrow(d), "; covariance=", round(cov_value, 4),
    "; Pearson r=", round(correlation, 4),
    ". Correlation measures linear association, not causation."
  )
}

topic8 <- function() {
  d <- data[complete.cases(data[, c("study_hours", "score_complete")]), ]
  fit <- lm(score_complete ~ study_hours + attendance_pct, data = d)
  sm <- summary(fit)
  cf <- coef(fit)
  p <- sm$coefficients[, 4]
  save_plot(8, function() {
    plot(fitted(fit), d$score_complete, pch = 19, col = "#33669988",
         xlab = "Fitted score", ylab = "Observed score",
         main = "Observed versus fitted score")
    abline(0, 1, col = "red", lty = 2)
  )
  paste0(
    "n=", nrow(d), "; equation: score=", round(cf[[1]], 3),
    " + ", round(cf[["study_hours"]], 3), "*study_hours + ",
    round(cf[["attendance_pct"]], 3), "*attendance; p-values (intercept, ",
    "hours, attendance)=(", paste(signif(p, 4), collapse = ", "),
    "); R^2=", round(sm$r.squared, 4),
    "; residual SD=", round(sigma(fit), 3),
    ". Slopes are adjusted associations."
  )
}

topic9 <- function() {
  a <- matrix(c(2, 1, 1, 3), nrow = 2, byrow = TRUE)
  b <- matrix(c(1, 4, 2, 0), nrow = 2, byrow = TRUE)
  rhs <- c(5, 7)
  solution <- solve(a, rhs)
  d <- data[complete.cases(data[, c("study_hours", "score_complete")]), ]
  x <- cbind(1, d$study_hours, d$score_complete)
  beta <- solve(crossprod(x), crossprod(x, d$attendance_pct))
  paste0(
    "A+B=", paste(as.vector(a + b), collapse = ","),
    "; A*B=", paste(as.vector(a %*% b), collapse = ","),
    "; transpose(A)=", paste(as.vector(t(a)), collapse = ","),
    "; det(A)=", det(a), "; inverse(A)=",
    paste(as.vector(solve(a)), collapse = ","),
    "; solution of A x=b=",
    paste(round(solution, 4), collapse = ","),
    "; residual norm=", format(sqrt(sum((a %*% solution - rhs)^2)), scientific = TRUE),
    ". Regression normal-equation coefficients (intercept, hours, score for ",
    "attendance)=", paste(round(beta, 4), collapse = ", "), "."
  )
}

topic10 <- function() {
  x <- data[complete.cases(data[, c("study_hours", "score_complete")]),
            c("study_hours", "attendance_pct", "score_complete")]
  pca <- prcomp(x, center = TRUE, scale. = TRUE)
  ratios <- pca$sdev^2 / sum(pca$sdev^2)
  save_plot(10, function() {
    plot(seq_along(ratios), ratios, type = "b", pch = 19,
         xlab = "Principal component", ylab = "Explained variance ratio",
         main = "PCA scree plot", xaxt = "n")
    axis(1, at = seq_along(ratios))
  )
  paste0(
    "Explained variance ratios=", paste(round(ratios, 4), collapse = ", "),
    "; eigenvalues=", paste(round(pca$sdev^2, 4), collapse = ", "),
    "; eigenvectors/components (columns)=",
    paste(capture.output(print(round(pca$rotation, 4))), collapse = "\n"),
    ". Standardization prevents original units from dominating PCA."
  )
}

topic11 <- function() {
  d <- data[complete.cases(data[, c("study_hours", "score_complete")]), ]
  objective <- function(beta) {
    sum((d$score_complete - beta[[1]] - beta[[2]] * d$study_hours)^2)
  }
  opt <- optim(c(0, 0), objective, method = "BFGS")
  closed <- coef(lm(score_complete ~ study_hours, data = d))
  save_plot(11, function() {
    plot(d$study_hours, d$score_complete, pch = 19, col = "#33669988",
         xlab = "Study hours", ylab = "Score",
         main = "Least-squares optimization")
    abline(opt$par[[1]], opt$par[[2]], col = "red", lwd = 2)
  )
  paste0(
    "Converged=", opt$convergence == 0, "; message=", opt$message,
    "; intercept=", round(opt$par[[1]], 4),
    "; slope=", round(opt$par[[2]], 4),
    "; sum of squared errors=", round(opt$value, 3),
    "; closed-form intercept/slope=(",
    paste(round(closed, 4), collapse = ", "), ")."
  )
}

topic12 <- function() {
  n <- 50000
  points <- matrix(runif(2 * n), ncol = 2)
  inside <- points[, 1]^2 + points[, 2]^2 <= 1
  estimate <- 4 * mean(inside)
  se <- 4 * sqrt(mean(inside) * (1 - mean(inside)) / n)
  estimates <- 4 * cumsum(inside) / seq_len(n)
  save_plot(12, function() {
    plot(seq_len(n), estimates, type = "l", lwd = 0.6,
         xlab = "Draws", ylab = "Estimate",
         main = "Monte Carlo approximation of pi")
    abline(h = pi, col = "red", lty = 2)
  )
  paste0("N=", n, "; estimate pi=", round(estimate, 6),
         "; approximate MC SE=", round(se, 6), ".")
}

topic13 <- function() {
  d <- data
  d$date <- as.Date(d$date)
  d$month <- format(d$date, "%Y-%m")
  monthly <- aggregate(
    d$score_complete, by = list(month = d$month), FUN = mean, na.rm = TRUE
  )
  names(monthly) <- c("month", "score_complete")
  time <- seq_len(nrow(monthly)) - 1
  fit <- lm(monthly$score_complete ~ time)
  forecast <- predict(fit, newdata = data.frame(time = nrow(monthly)))
  acf <- cor(monthly$score_complete[-1], monthly$score_complete[-nrow(monthly)])
  monthly$month_number <- as.integer(substr(monthly$month, 6, 7))
  seasonal <- aggregate(
    monthly$score_complete, by = list(month = monthly$month_number), FUN = mean
  )$x
  trend_interpretation <- if (summary(fit)$coefficients[2, 4] < 0.05) {
    "The linear trend is statistically detectable at alpha=0.05."
  } else {
    "The linear trend is not statistically detectable at alpha=0.05."
  }
  dates <- as.Date(paste0(monthly$month, "-01"))
  save_plot(13, function() {
    plot(dates, monthly$score_complete, type = "b", pch = 19,
         xlab = "Month", ylab = "Mean score",
         main = "Monthly mean complete score")
    lines(dates, fitted(fit), lty = 2, col = "red")
  )
  paste0(
    "Monthly observations=", nrow(monthly), "; first=", monthly$month[[1]],
    ", last=", monthly$month[[nrow(monthly)]],
    "; trend=", round(coef(fit)[[2]], 4),
    " score/month (p=", signif(summary(fit)$coefficients[2, 4], 4),
    "); lag-1 autocorrelation=", round(acf, 4),
    "; calendar-month mean-score range=(",
    paste(round(range(seasonal), 2), collapse = ", "), ")",
    "; next-month linear-trend forecast=", round(forecast, 2),
    ". ", trend_interpretation, " Forecast is illustrative."
  )
}

topic14 <- function() {
  d <- data
  d$date <- as.Date(d$date)
  d$month <- format(d$date, "%Y-%m")
  monthly <- aggregate(
    d$score_complete, by = list(month = d$month), FUN = mean, na.rm = TRUE
  )
  names(monthly) <- c("month", "mean")
  save_plot(14, function() {
    oldpar <- par(mfrow = c(2, 2))
    hist(d$score_complete, breaks = 18, main = "Score histogram", xlab = "Score")
    boxplot(score_complete ~ group, data = d, main = "Scores by group",
            xlab = "Group", ylab = "Score")
    plot(d$study_hours, d$score_complete, pch = 19, col = "#33669988",
         main = "Study hours vs score", xlab = "Hours", ylab = "Score")
    plot(as.Date(paste0(monthly$month, "-01")), monthly$mean, type = "b",
         main = "Monthly mean score", xlab = "Month", ylab = "Mean score")
    par(oldpar)
  }, width = 10, height = 7)
  "Created labeled histogram, group boxplot, scatterplot, and time-series line chart."
}

topic15 <- function() {
  x <- data$score_complete
  z <- as.numeric(scale(x))
  scaled <- (x - min(x)) / (max(x) - min(x))
  save_plot(15, function() {
    oldpar <- par(mfrow = c(1, 3))
    hist(x, breaks = 18, col = "steelblue",
         main = "Original", xlab = "Score")
    hist(z, breaks = 18, col = "darkorange",
         main = "Z-score", xlab = "Standardized value")
    hist(scaled, breaks = 18, col = "seagreen",
         main = "Min-max", xlab = "Scaled value")
    par(oldpar)
  )
  paste0(
    "Original mean/SD=(", round(mean(x), 3), ", ", round(sd(x), 3),
    "); z-score mean/SD=(", signif(mean(z), 3), ", ", round(sd(z), 3),
    "); min-max range=(", min(scaled), ", ", max(scaled), ")."
  )
}

topic16 <- function() {
  d <- data[!is.na(data$score_complete), ]
  fit <- aov(score_complete ~ group, data = d)
  table <- summary(fit)[[1]]
  tukey <- TukeyHSD(fit)
  means <- tapply(d$score_complete, d$group, mean)
  interpretation <- if (table$`Pr(>F)`[[1]] < 0.05) {
    "At alpha=0.05, the omnibus test indicates group means differ."
  } else {
    "At alpha=0.05, the omnibus test does not detect a group-mean difference."
  }
  paste0(
    "Group means=", paste(names(means), round(means, 3), sep = ":", collapse = ", "),
    "; F(", table$Df[[1]], ",", table$Df[[2]], ")=",
    round(table$`F value`[[1]], 4), "; p=",
    signif(table$`Pr(>F)`[[1]], 5), "; ANOVA table:\n",
    paste(capture.output(print(table)), collapse = "\n"),
    "\nTukey HSD:\n",
    paste(capture.output(print(tukey)), collapse = "\n"),
    "\n", interpretation,
    " Check assumptions and note that group assignment was not randomized."
  )
}

topic17 <- function() {
  d <- data[!is.na(data$passed), ]
  tab <- table(d$group, d$passed)
  test <- chisq.test(tab, correct = FALSE)
  interpretation <- if (test$p.value < 0.05) {
    "Evidence of an association at alpha=0.05."
  } else {
    "No statistically significant association detected at alpha=0.05."
  }
  paste0(
    "Observed table (columns 0=not pass, 1=pass):\n",
    paste(capture.output(print(tab)), collapse = "\n"),
    "\nExpected counts:\n",
    paste(capture.output(print(round(test$expected, 2))), collapse = "\n"),
    "\nchi-square(", unname(test$parameter), ")=",
    round(unname(test$statistic), 4), "; p=",
    signif(test$p.value, 5), ". ", interpretation,
    " Assess expected counts; an association is not causal."
  )
}

topic18 <- function() {
  d <- data[complete.cases(data[, c("study_hours", "attendance_pct", "passed")]), ]
  fit <- glm(passed ~ study_hours + attendance_pct, data = d, family = binomial())
  prediction <- as.integer(predict(fit, type = "response") >= 0.5)
  representative_probability <- predict(
    fit,
    newdata = data.frame(
      study_hours = median(d$study_hours),
      attendance_pct = median(d$attendance_pct)
    ),
    type = "response"
  )
  truth <- as.integer(d$passed)
  cm <- table(factor(truth, levels = 0:1), factor(prediction, levels = 0:1))
  tp <- cm[2, 2]
  precision <- if (sum(cm[, 2]) == 0) 0 else tp / sum(cm[, 2])
  recall <- if (sum(cm[2, ]) == 0) 0 else tp / sum(cm[2, ])
  f1 <- if (precision + recall == 0) 0 else 2 * precision * recall / (precision + recall)
  paste0(
    "Coefficients (intercept, hours, attendance)=",
    paste(round(coef(fit), 4), collapse = ", "),
    "; odds ratios=", paste(round(exp(coef(fit)), 4), collapse = ", "),
    "; predicted pass probability at median predictors=",
    round(representative_probability, 4),
    "; apparent confusion matrix (rows/columns 0,1):\n",
    paste(capture.output(print(cm)), collapse = "\n"),
    "\naccuracy=", round(mean(prediction == truth), 3),
    "; precision=", round(precision, 3), "; recall=", round(recall, 3),
    "; F1=", round(f1, 3),
    ". Training-set performance is optimistic; validate on held-out data."
  )
}

topic19 <- function() {
  female <- data$score_complete[data$gender == "Female"]
  male <- data$score_complete[data$gender == "Male"]
  rank_test <- wilcox.test(female, male, exact = FALSE)
  t_test <- t.test(female, male)
  interpretation <- if (rank_test$p.value < 0.05) {
    "The rank test detects a distributional difference at alpha=0.05."
  } else {
    "The rank test does not detect a distributional difference at alpha=0.05."
  }
  paste0(
    "Female n=", length(female), ", median=", round(median(female), 2),
    "; male n=", length(male), ", median=", round(median(male), 2),
    "; Mann-Whitney W=", unname(rank_test$statistic),
    ", p=", signif(rank_test$p.value, 5),
    "; Welch t=", round(unname(t_test$statistic), 4),
    ", p=", signif(t_test$p.value, 5),
    ". ", interpretation,
    " The rank test compares distributions, not necessarily medians alone."
  )
}

topic20 <- function() {
  x <- data$score_complete
  boot <- replicate(5000, mean(sample(x, length(x), replace = TRUE)))
  ci <- quantile(boot, c(0.025, 0.975))
  save_plot(20, function() {
    hist(boot, breaks = 35, col = "grey80",
         main = "Bootstrap sampling distribution of the mean",
         xlab = "Bootstrap mean")
    abline(v = mean(x), col = "red", lty = 2)
  )
  paste0(
    "Resamples=5,000; bootstrap mean=", round(mean(boot), 3),
    "; bootstrap SE=", round(sd(boot), 3),
    "; percentile 95% CI=(", round(ci[[1]], 3), ", ",
    round(ci[[2]], 3), "). Bootstrap cannot remove sampling bias."
  )
}

topic21 <- function() {
  x <- data[complete.cases(data[, c("study_hours", "score_complete")]),
            c("study_hours", "attendance_pct", "score_complete")]
  z <- scale(x)
  set.seed(4131)
  fit <- kmeans(z, centers = 3, nstart = 10)
  paste0(
    "Variables=(study_hours, attendance_pct, score_complete); correlation matrix:\n",
    paste(capture.output(print(round(cor(x), 3))), collapse = "\n"),
    "\nCovariance matrix:\n",
    paste(capture.output(print(round(cov(x), 3))), collapse = "\n"),
    "\nStandardized k-means cluster counts=",
    paste(fit$size, collapse = ", "),
    ". Cluster IDs are arbitrary and depend on scaling and initialization."
  )
}

topic22 <- function() {
  columns <- c("study_hours", "attendance_pct", "score")
  counts <- colSums(is.na(data[, columns]))
  save_plot(22, function() {
    image(
      x = seq_len(nrow(data)), y = seq_along(columns),
      z = is.na(as.matrix(data[, columns])) * 1,
      col = c("white", "black"), axes = FALSE,
      xlab = "Record", ylab = "", main = "Missingness map (black = missing)"
    )
    axis(2, at = seq_along(columns), labels = columns, las = 1)
    axis(1)
  }, width = 9, height = 4)
  imputed <- data[, columns]
  for (column in columns) {
    imputed[[column]][is.na(imputed[[column]])] <-
      median(imputed[[column]], na.rm = TRUE)
  }
  paste0(
    "Missing counts: ", paste(names(counts), counts, sep = "=", collapse = ", "),
    "; median-imputed score mean=", round(mean(imputed$score), 3),
    "; observed score mean=", round(mean(data$score, na.rm = TRUE), 3),
    ". Median imputation is illustrative and understates uncertainty."
  )
}

topic23 <- function() {
  paste0(
    "This runner records methods and generated results, uses fixed random seeds, ",
    "reads the shared CSV, and saves figures alongside this report. Include ",
    "the report, topic guide, and both runners in the group's GitHub repository."
  )
}

topic_functions <- list(
  topic1, topic2, topic3, topic4, topic5, topic6, topic7, topic8,
  topic9, topic10, topic11, topic12, topic13, topic14, topic15,
  topic16, topic17, topic18, topic19, topic20, topic21, topic22, topic23
)
results <- character(length(topic_functions))
figure_topics <- c(1, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 20, 22)
for (i in seq_along(topic_functions)) {
  result <- topic_functions[[i]]()
  image <- if (i %in% figure_topics) {
    paste0("\n\n![Topic ", i, " visualization](figures/topic_",
           sprintf("%02d", i), ".png)\n")
  } else {
    ""
  }
  results[[i]] <- paste0(
    "## ", i, ". ", topic_titles[[i]], "\n\n", result, image, "\n"
  )
  cat("\n[", sprintf("%02d", i), "] ", topic_titles[[i]], "\n", result, "\n", sep = "")
}

report <- paste0(
  "# BIT 4131 reproducible results (R)\n\n",
  "Dataset: [`student_performance.csv`](../../data/student_performance.csv) (",
  nrow(data), " synthetic records). Results and figures were generated by ",
  "[`run_all.R`](../../R/run_all.R); see the ",
  "[topic guide](../../docs/topic_guide.md) for assumptions, theory, and equations. ",
  "These are educational examples, not real-student findings.\n\n",
  paste(results, collapse = "\n"),
  "\n## Overall conclusion and limitations\n\n",
  "In this generated dataset, study hours and attendance are positively ",
  "associated with assessment score, and the fitted models summarize those ",
  "relationships. Group-level tests provide mixed evidence, illustrating why ",
  "effect estimates, assumptions, and post-hoc comparisons matter. All records ",
  "are synthetic, models are demonstrations rather than causal claims, and ",
  "training-sample classification metrics are not out-of-sample estimates.\n"
)
writeLines(report, file.path(out_dir, "report.md"), useBytes = TRUE)
cat("\nReport written to results/R/report.md\n")
