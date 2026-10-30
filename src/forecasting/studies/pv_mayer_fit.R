# Fit censored Gaussian baselines and evaluate scores in the isolated R environment.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop('Expected training CSV, prediction CSV and output directory')
if (!requireNamespace('crch', quietly = TRUE) || !requireNamespace('scoringRules', quietly = TRUE)) stop('crch and scoringRules are required')
train <- read.csv(args[1], check.names = FALSE)
test <- read.csv(args[2], check.names = FALSE)
out <- args[3]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
writeLines(c(paste('R', getRversion()), paste('crch', packageVersion('crch')),
             paste('scoringRules', packageVersion('scoringRules')),
             paste('crch.boost defaults:', paste(capture.output(print(crch::crch.boost())), collapse=' '))),
           file.path(out, 'r_versions_and_defaults.txt'))
predictions <- list()
status <- list()
fit_one <- function(method, lead, fit_data, score_data) {
  name <- if (is.na(lead)) method else paste0(method, '_lead_', lead)
  tryCatch({
    if (!nrow(fit_data)) stop('No eligible fit rows')
    if (method == 'cn_emos') {
      model <- crch::crch(y ~ control + ens_mean | log_variance, data=fit_data,
                          left=0, right=1, dist='gaussian', link.scale='log', type='crps')
    } else {
      model <- crch::crch(y ~ control + ens_mean + ens_sd | control + ens_mean + ens_sd,
                          data=fit_data, left=0, right=1, dist='gaussian', link.scale='log',
                          type='ml', control=crch::crch.boost())
    }
    if (!is.null(model$converged) && !isTRUE(model$converged)) stop('R optimizer did not converge')
    saveRDS(model, file.path(out, paste0(name, '.rds')))
    if (nrow(score_data)) {
      location <- as.numeric(predict(model, newdata=score_data, type='location'))
      scale <- as.numeric(predict(model, newdata=score_data, type='scale'))
      good <- is.finite(location) & is.finite(scale) & scale > 0
      crps <- rep(NA_real_, length(scale))
      crps[good] <- scoringRules::crps_cnorm(score_data$y[good], location[good], scale[good], lower=0, upper=1)
      predictions[[length(predictions)+1L]] <<- data.frame(row_id=score_data$row_id, method=method,
                                                           location=location, scale=scale, crps=crps)
    }
    status[[length(status)+1L]] <<- data.frame(method=method, lead=lead, fit_rows=nrow(fit_data), status='complete', detail='')
  }, error=function(e) {
    status[[length(status)+1L]] <<- data.frame(method=method, lead=lead, fit_rows=nrow(fit_data), status='failed', detail=conditionMessage(e))
  })
}
for (lead in sort(unique(c(train$lead, test$lead)))) {
  fit_one('cn_emos', lead, train[train$lead == lead & train$variance > 0, ],
          test[test$lead == lead & test$variance > 0, ])
}
fit_one('cn_emos_b', NA_real_, train, test)
result <- if (length(predictions)) do.call(rbind, predictions) else data.frame(row_id=integer(), method=character(), location=numeric(), scale=numeric(), crps=numeric())
write.csv(result, file.path(out, 'predictions.csv'), row.names=FALSE, na='')
write.csv(do.call(rbind, status), file.path(out, 'fit_status.csv'), row.names=FALSE, na='')
