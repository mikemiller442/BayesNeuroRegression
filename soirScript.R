








### Load libraries
library(BayesGPfit)
library(ggplot2)
library(rgl)
library(glmnet)
library(viridis)
library(lme4)
library(Rcpp)
library(plotly)
library(geometry)

### set RNG seed
set.seed(80924)

generate_sphere_coords <- function(n_points) {
  # Spherical Fibonacci point set
  golden_ratio <- (1+sqrt(5))/2
  i <- seq(0,n_points-1)
  
  z <- 1-(2*i+1)/n_points
  radius <- sqrt(1-z^2)
  
  phi <- 2*pi*i/golden_ratio
  
  x <- radius*cos(phi)
  y <- radius*sin(phi)
  
  coords <- cbind(x,y,z)
  
  return(coords)
}

### Generate synthetic brain activation data using Gaussian Process with independent noise
generate_synthetic_activation <- function(gp_coords, n_subjects, a = 0.001, b = 20.0, 
                                          noise_sd = 0.1, add_mean_shift = TRUE,
                                          mean_shift = 0.5) {
  
  n_vertices <- nrow(gp_coords)
  norms_sq <- rowSums(gp_coords^2)
  
  # Vectorized computation
  dist_matrix_sq <- as.matrix(dist(gp_coords, method = "euclidean"))^2
  norms_outer_sum <- outer(norms_sq, norms_sq, "+")
  K <- exp(-a*norms_outer_sum - b*dist_matrix_sq)
  K <- K + diag(1e-6, n_vertices)
  L <- chol(K)
  
  Z <- matrix(rnorm(n_subjects * n_vertices), nrow = n_vertices, ncol = n_subjects)
  noise_matrix <- matrix(rnorm(n_subjects * n_vertices, 0, noise_sd), 
                         nrow = n_vertices, ncol = n_subjects)
  
  # Vectorized multiplication: each column of Z is a sample
  gp_samples <- t(L) %*% Z
  activation_matrix <- t(gp_samples + noise_matrix)
  
  # Add mean shift to covariates to create non-zero mean
  if (add_mean_shift) {
    # Add spatially varying mean (stronger on right hemisphere)
    hemisphere_weights <- ifelse(gp_coords[,1] > 0, 1.0, 0.8)
    for (i in 1:n_subjects) {
      activation_matrix[i,] <- activation_matrix[i,] + mean_shift * hemisphere_weights
    }
    cat("Mean shift added to covariates:", mean_shift, "\n")
    cat("Covariate mean:", mean(activation_matrix), "\n")
    cat("Covariate SD:", sd(as.vector(activation_matrix)), "\n\n")
  }
  
  return(activation_matrix)
}

### Generate regression coefficients using Gaussian Process with lambdaSq scaling
generate_gp_coefficients_scaled <- function(gp_coords, a_coef = 0.01, b_coef = 5.0, 
                                            target_sd = 0.0001) {
  
  n_vertices <- nrow(gp_coords)
  norms_sq <- rowSums(gp_coords^2)
  
  # compute expected variance from the kernel to determine lambdaSq
  # given K(x,x') = lambdaSq * exp(-a*(||x||^2 + ||x'||^2) - b*||x-x'||^2)
  # the marginal variance at point x is Var(f(x)) = K(x,x) = lambdaSq * exp(-2*a*||x||^2)
  
  # Calculate expected variances at each point (with lambdaSq = 1)
  expected_variances_base <- exp(-2*a_coef*norms_sq)
  expected_sd_base <- sqrt(mean(expected_variances_base))
  
  # Calculate lambdaSq to achieve target SD
  lambdaSq <- (target_sd/expected_sd_base)^2
  
  # Vectorized computation of covariance matrix
  # Compute all pairwise squared distances at once
  dist_matrix_sq <- as.matrix(dist(gp_coords, method = "euclidean"))^2
  
  # Compute the outer sum of norms_sq
  norms_outer_sum <- outer(norms_sq, norms_sq, "+")
  
  # Vectorized covariance matrix computation
  K <- lambdaSq*exp(-a_coef*norms_outer_sum - b_coef*dist_matrix_sq)
  K <- K + diag(1e-9, n_vertices)
  
  # Generate sample
  L <- chol(K)
  z <- rnorm(n_vertices)
  gp_sample <- t(L) %*% z
  coefficients <- as.vector(gp_sample)
  
  return(list(
    coefficients = coefficients,
    lambdaSq = lambdaSq,
    a_coef = a_coef,
    b_coef = b_coef
  ))
}

### Generate synthetic brain-like data for whole sphere
n_vertices_total <- 6000

### Parameters for data generation
a_data <- 0.001
b_data <- 20.0

# Generate coefficients with fixed a=0.001, b=5.0, and lambdaSq scaling
# No mean shift needed for coefficients since covariates now have non-zero mean
target_sd <- 0.001

### read generated data
sphere_coord <- readRDS("./sphere_coord.rds")
cortex_dat <- readRDS("./cortex_dat.rds")
coef_result <- readRDS("./coef_result.rds")

# # Generate coordinates for entire sphere (use Cartesian directly)
# sphere_coord <- generate_sphere_coords(n_vertices_total)
# ### Generate covariates and coefficients for entire sphere
# # Add mean shift to covariates to create non-zero imaging signal
# cortex_dat <- generate_synthetic_activation(sphere_coord, n_subjects = 3000,
#                                             a = a_data, b = b_data, add_mean_shift = TRUE,
#                                             mean_shift = 0.5)
# coef_result <- generate_gp_coefficients_scaled(sphere_coord, a_coef = 0.001, b_coef = 5.0,
#                                                target_sd = target_sd)

# ### save generated data
# saveRDS(sphere_coord,"./sphere_coord.rds")
# saveRDS(cortex_dat,"./cortex_dat.rds")
# saveRDS(coef_result,"./coef_result.rds")

# initialize GP and regression model parameters
beta_true <- coef_result$coefficients
lambdaSq_true <- coef_result$lambdaSq
a_coef_true <- coef_result$a_coef
b_coef_true <- coef_result$b_coef
mean_shift_true <- coef_result$mean_shift

### Generate response variable with hierarchical structure
intercept_global <- 0.0
beta_continuous_mean <- 1.0
noise_sd_response <- 1.0
n_subjects <- nrow(cortex_dat)

# Create cluster variable with 20 unique IDs
n_clusters <- 20
cluster_id <- rep(1:n_clusters, length.out = n_subjects)
cluster_id <- sample(cluster_id)

# Generate continuous covariate
x_continuous <- rnorm(n_subjects, mean = 0, sd = 1)

# Generate random effects for each cluster
sd_intercept <- 0.1
random_intercepts <- rnorm(n_clusters, mean = 0, sd = sd_intercept)

sd_slope <- 0.1
random_slopes <- rnorm(n_clusters, mean = 0.5, sd = sd_slope)

# Generate response variable
y <- rep(0, n_subjects)

for (i in 1:n_subjects) {
  cluster <- cluster_id[i]
  
  fixed_effects <- intercept_global + beta_continuous_mean * x_continuous[i]
  random_effects <- random_intercepts[cluster] + random_slopes[cluster] * x_continuous[i]
  image_effects <- sum(cortex_dat[i,] * beta_true)
  
  y[i] <- fixed_effects + random_effects + image_effects
}

y <- y + rnorm(n_subjects, 0, noise_sd_response)

# residualize response variable
mixed_data <- data.frame(
  y = y,
  x_continuous = x_continuous,
  cluster_id = as.factor(cluster_id)
)

lmer_model <- lmer(y ~ x_continuous + (1 + x_continuous | cluster_id), 
                   data = mixed_data,
                   REML = TRUE)

y_predicted_lmer <- predict(lmer_model)
y_residualized <- y - y_predicted_lmer

### Function to fit ridge regression model with fixed basis parameters (3D Cartesian)
fit_ridge_with_basis_3d <- function(a_basis, b_basis, cortex, response,
                                    coords_3d, poly_degree,
                                    alpha_param = 0.0, nfolds = 10) {
  
  # Calculate basis functions for 3D Cartesian space
  # Note: d = 3 for 3D coordinates
  Psi <- GP.eigen.funcs.fast(coords_3d, poly_degree = poly_degree, a = a_basis, b = b_basis)
  lambda <- GP.eigen.value(poly_degree = poly_degree, a = a_basis, b = b_basis, d = ncol(coords_3d))
  
  # Transform to basis representation
  imagingFeatures <- cortex %*% t(t(Psi) * sqrt(lambda))
  
  # Fit ridge regression with cross-validation (alpha = 0.0)
  cv_model <- cv.glmnet(x = imagingFeatures, 
                        y = response,
                        alpha = alpha_param,
                        nfolds = nfolds,
                        standardize = TRUE,
                        type.measure = "mse")
  
  rmse <- sqrt(min(cv_model$cvm))
  
  return(list(
    rmse = rmse,
    cv_model = cv_model,
    Psi = Psi,
    lambda = lambda,
    imagingFeatures = imagingFeatures
  ))
}

### Use fixed basis parameters (no cross-validation)
# Use same a and b values as coefficient generation
a_basis <- 0.001
b_basis <- 5.0
poly_degree <- 20

# Calculate expected number of basis functions
n_basis_expected <- choose(poly_degree + 3, 3)
cat("  Expected number of basis functions:", n_basis_expected, "\n\n")

### Fit model with fixed parameters
cat("\n=== Fitting Ridge Regression Model (alpha = 0.0) ===\n")

final_fit <- fit_ridge_with_basis_3d(
  a_basis = a_basis,
  b_basis = b_basis,
  cortex = cortex_dat,
  response = y_residualized,
  coords_3d = sphere_coord,
  poly_degree = poly_degree
)

cat("Number of basis functions used:", ncol(final_fit$imagingFeatures), "\n\n")

# Get final model
alpha_param <- 0.0
final_model <- glmnet(x = final_fit$imagingFeatures,
                      y = y_residualized,
                      alpha = alpha_param,
                      lambda = final_fit$cv_model$lambda.min,
                      standardize = TRUE)

predictions <- predict(final_model, newx = final_fit$imagingFeatures, 
                       s = final_fit$cv_model$lambda.min)

mse <- mean((y_residualized - predictions)^2)
rmse <- sqrt(mse)
r_squared <- cor(y_residualized, predictions)^2

# Get coefficients in basis space
beta_basis <- coef(final_model, s = final_fit$cv_model$lambda.min)
beta_basis_vec <- as.vector(beta_basis[-1])

# Transform back to vertex space
beta_estimated <- t(t(final_fit$Psi) * sqrt(final_fit$lambda)) %*% beta_basis_vec

correlation_total <- cor(beta_estimated, beta_true)

cat("Final model performance:\n")
cat("  R-squared:", r_squared, "\n")
cat("  RMSE:", rmse, "\n")
cat("  Correlation (whole sphere):", correlation_total, "\n\n")


### Function to create smooth surface visualization with plotly
visualize_sphere_smooth <- function(coords, activation, 
                                    title = "Brain Activation",
                                    color_palette = "Plasma") {
  
  # Use convhulln to get surface triangles directly
  hull_result <- convhulln(coords, options = "FA")
  triangles <- hull_result$hull
  
  # Convert to 0-indexed for plotly (plotly uses 0-indexing)
  i <- triangles[, 1] - 1
  j <- triangles[, 2] - 1
  k <- triangles[, 3] - 1
  
  # Create the 3D mesh plot
  fig <- plot_ly(x = coords[, 1],y = coords[, 2],z = coords[, 3],
    i = i,j = j,k = k,
    intensity = activation,
    type = "mesh3d",
    colorscale = color_palette,
    showscale = TRUE,
    colorbar = list(title = "Coefficient"),
    hoverinfo = "text",
    text = paste("Value:", round(activation, 6))
  )
  
  # Update layout
  fig <- fig %>% layout(
    title = title,
    scene = list(
      xaxis = list(title = "X"),
      yaxis = list(title = "Y"),
      zaxis = list(title = "Z"),
      aspectmode = "data"
    )
  )
  
  return(fig)
}

### Visualize True Coefficients - Whole Sphere
cat("Creating smooth visualization for True Coefficients...\n")
fig_true <- visualize_sphere_smooth(
  coords = sphere_coord,
  activation = beta_true,
  title = "TRUE Coefficients - Whole Sphere (Smooth)",
  color_palette = "Plasma"
)

print(fig_true)

### Visualize Ridge Regression Coefficients - Whole Sphere
cat("Creating smooth visualization for Ridge Regression Coefficients...\n")
fig_ridge <- visualize_sphere_smooth(
  coords = sphere_coord,
  activation = beta_estimated,
  title = "RIDGE Coefficients - Whole Sphere (Smooth)",
  color_palette = "Plasma"
)

print(fig_ridge)

### Side-by-side comparison: True vs Ridge
cat("Creating side-by-side comparison...\n")

# Offset coordinates for side-by-side comparison
offset <- 2.5

# True coefficients (left position)
coords_true <- sphere_coord
coords_true[, 1] <- sphere_coord[, 1] - offset

# Ridge coefficients (right position)
coords_ridge <- sphere_coord
coords_ridge[, 1] <- sphere_coord[, 1] + offset

# Create meshes
fig_comparison <- plot_ly()

# True coefficients
hull_true <- convhulln(coords_true, options = "FA")
tri_true <- hull_true$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_true[, 1],y = coords_true[, 2],z = coords_true[, 3],
  i = tri_true[, 1] - 1,j = tri_true[, 2] - 1,k = tri_true[, 3] - 1,
  intensity = beta_true,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = TRUE,
  colorbar = list(title = "Coefficient", x = 1.1),
  name = "True",
  hoverinfo = "text",
  text = paste("True:", round(beta_true, 6))
)

# Ridge coefficients
hull_ridge <- convhulln(coords_ridge, options = "FA")
tri_ridge <- hull_ridge$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_ridge[, 1],y = coords_ridge[, 2],z = coords_ridge[, 3],
  i = tri_ridge[, 1] - 1,j = tri_ridge[, 2] - 1,k = tri_ridge[, 3] - 1,
  intensity = beta_estimated,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = FALSE,
  name = "Ridge",
  hoverinfo = "text",
  text = paste("Ridge:", round(beta_estimated, 6))
)

# Update layout with annotations
fig_comparison <- fig_comparison %>% layout(
  title = "Comparison: True vs Ridge Coefficients",
  scene = list(
    xaxis = list(title = "X"),
    yaxis = list(title = "Y"),
    zaxis = list(title = "Z"),
    aspectmode = "data",
    camera = list(eye = list(x = 0, y = -2.5, z = 0.3)),
    annotations = list(
      list(x = -offset, y = 0, z = 1.5, text = "TRUE", showarrow = FALSE),
      list(x = offset, y = 0, z = 1.5, text = "RIDGE", showarrow = FALSE)
    )
  )
)

print(fig_comparison)











### Bayesian regression
sourceCpp("robustLinRegMMGS.cpp")

y_centered <- y - mean(y)
x_continuous_centered <- x_continuous - mean(x_continuous)
modelControl <- cbind(rep(1, n_subjects), x_continuous_centered)
imagingFeatures <- final_fit$imagingFeatures
groupID <- cluster_id - 1

numEpochs <- 6000
numDiscard <- 1500
paramPeriod <- 100
lambdaSqStart <- lambdaSq_true
SigmaStart <- 1e-2
numGroups <- n_clusters

# res <- linRegMMGS(Z = modelControl,
#                   testZ = modelControl,
#                   X = imagingFeatures,
#                   testX = imagingFeatures,
#                   Y = y_centered,
#                   testY = y_centered,
#                   groupVec = groupID,
#                   testGroupVec = groupID,
#                   mixedX = modelControl[,c(1,2)],
#                   testMixedX = modelControl[,c(1,2)],
#                   lambdaSqStart = lambdaSqStart,
#                   SigmaStart = SigmaStart,
#                   numGroups = numGroups,
#                   numEpochs = numEpochs,
#                   numDiscard = numDiscard,
#                   paramPeriod = paramPeriod)

# ### save Bayesian scalar on image regression model
# saveRDS(res,"./soirModel.rds")

### read Bayesian scalar on image regression model
res <- readRDS("./soirModel.rds")

### model diagnostics
postControlBeta <- res$controlBeta[,(numDiscard+2):(numEpochs+1)]
postNeuroBeta <- res$neuroBeta[,(numDiscard+2):(numEpochs+1)]
postLambdaSq <- res$lambdaSq[(numDiscard+2):(numEpochs+1)]
postLambdaSqScale <- res$lambdaSqScale[(numDiscard+2):(numEpochs+1)]
postRegVar <- res$regVar[(numDiscard+2):(numEpochs+1)]
postRegVarScale <- res$regVarScale[(numDiscard+2):(numEpochs+1)]
postReVar <- res$reVar[,(numDiscard+2):(numEpochs+1)]


trainMSE = mean((y_centered - res$trainPostPred)^2)
print(paste0("Train MSE: ", trainMSE))


trainMSE = mean((y_centered - res$trainPartialPostPred)^2)
print(paste0("Train MSE: ", trainMSE))


meIdx <- 1
slopeIdx <- 1


trace_df <- data.frame(iter = 1:length(postControlBeta[meIdx,]),
                       intercept = postControlBeta[meIdx,])
ggplot(trace_df, aes(x=iter,y=intercept)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postControlBeta[meIdx,]), color = "red") +
  ggtitle("Intercept Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))


trace_df <- data.frame(iter = 1:length(postNeuroBeta[slopeIdx,]),
                       intercept = postNeuroBeta[slopeIdx,])
ggplot(trace_df, aes(x=iter,y=intercept)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postNeuroBeta[slopeIdx,]), color = "red") +
  ggtitle("Slope Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))


trace_df <- data.frame(iter = 1:length(postLambdaSq), lambdaSq = postLambdaSq)
ggplot(trace_df, aes(x=iter,y=lambdaSq)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postLambdaSq), color = "red") +
  ggtitle("LambdaSq Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))


hist_df <- data.frame(lambdaSq = postLambdaSq)
ggplot(hist_df, aes(x=lambdaSq)) + geom_histogram(bins = 100, color="darkblue", fill="lightblue") +
  ggtitle("LambdaSq Posterior Density") +
  theme(plot.title = element_text(hjust = 0.5))


trace_df <- data.frame(iter = 1:length(postRegVar), regVar = postRegVar)
ggplot(trace_df, aes(x=iter,y=regVar)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postRegVar), color = "red") +
  ggtitle("Reg Var Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))


hist_df <- data.frame(regVar = postRegVar)
ggplot(hist_df, aes(x=regVar)) + geom_histogram(bins = 100, color="darkblue", fill="lightblue") +
  ggtitle("Reg Var  Posterior Density") +
  theme(plot.title = element_text(hjust = 0.5))


trace_df <- data.frame(iter = 1:length(postReVar[1,]), reVar = postReVar[1,])
ggplot(trace_df, aes(x=iter,y=reVar)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postReVar[1,]), color = "red") +
  ggtitle("Random Effect Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))


trace_df <- data.frame(iter = 1:length(postReVar[2,]), reVar = postReVar[2,])
ggplot(trace_df, aes(x=iter,y=reVar)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postReVar[2,]), color = "red") +
  ggtitle("Random Effect Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))










### Transform Bayesian coefficients
beta_basis_bayes <- res$neuroBeta
beta_bayes <- colSums(t(final_fit$Psi %*% (diag(sqrt(final_fit$lambda)) %*% postNeuroBeta[,])))

correlation_bayes <- cor(beta_bayes, beta_true)
mse_bayes <- mean((beta_bayes - beta_true)^2)

cat("=== Comparing Bayesian vs True Coefficients ===\n")
cat("  Correlation:", correlation_bayes, "\n")
cat("  MSE:", mse_bayes, "\n")
cat("  RMSE:", sqrt(mse_bayes), "\n\n")

### Visualize Bayesian coefficients
cat("=== Visualizing Bayesian Coefficients ===\n")



### Function to create smooth surface visualization with plotly
visualize_sphere_smooth <- function(coords, activation, 
                                    title = "Brain Activation",
                                    color_palette = "Plasma") {
  
  # Use convhulln to get surface triangles directly
  hull_result <- convhulln(coords, options = "FA")
  triangles <- hull_result$hull
  
  # Convert to 0-indexed for plotly (plotly uses 0-indexing)
  i <- triangles[, 1] - 1
  j <- triangles[, 2] - 1
  k <- triangles[, 3] - 1
  
  # Create the 3D mesh plot
  fig <- plot_ly(
    x = coords[, 1],y = coords[, 2],z = coords[, 3],
    i = i,j = j,k = k,
    intensity = activation,
    type = "mesh3d",
    colorscale = color_palette,
    showscale = TRUE,
    colorbar = list(title = "Coefficient"),
    hoverinfo = "text",
    text = paste("Value:", round(activation, 6))
  )
  
  # Update layout
  fig <- fig %>% layout(
    title = title,
    scene = list(
      xaxis = list(title = "X"),
      yaxis = list(title = "Y"),
      zaxis = list(title = "Z"),
      aspectmode = "data"
    )
  )
  
  return(fig)
}

### Visualize True Coefficients - Whole Sphere
cat("Creating smooth visualization for True Coefficients...\n")
fig_true <- visualize_sphere_smooth(
  coords = sphere_coord,
  activation = beta_true,
  title = "TRUE Coefficients - Whole Sphere (Smooth)",
  color_palette = "Plasma"
)

print(fig_true)

### Visualize Bayesian Coefficients - Whole Sphere
cat("Creating smooth visualization for Bayesian Coefficients...\n")
fig_bayes <- visualize_sphere_smooth(
  coords = sphere_coord,
  # activation = beta_bayes,
  activation = beta_estimated,
  title = "BAYESIAN Coefficients - Whole Sphere (Smooth)",
  color_palette = "Plasma"
)

print(fig_bayes)

### Side-by-side comparison: True vs Bayesian
cat("Creating side-by-side comparison...\n")

# Offset coordinates for side-by-side comparison
offset <- 2.5

# True coefficients (left position)
coords_true <- sphere_coord
coords_true[, 1] <- sphere_coord[, 1] - offset

# Bayesian coefficients (right position)
coords_bayes <- sphere_coord
coords_bayes[, 1] <- sphere_coord[, 1] + offset

# Create meshes
fig_comparison <- plot_ly()

# True coefficients
hull_true <- convhulln(coords_true, options = "FA")
tri_true <- hull_true$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_true[, 1],y = coords_true[, 2],z = coords_true[, 3],
  i = tri_true[, 1] - 1,j = tri_true[, 2] - 1,k = tri_true[, 3] - 1,
  intensity = beta_true,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = TRUE,
  colorbar = list(title = "Coefficient", x = 1.1),
  name = "True",
  hoverinfo = "text",
  text = paste("True:", round(beta_true, 6))
)

# Bayesian coefficients
hull_bayes <- convhulln(coords_bayes, options = "FA")
tri_bayes <- hull_bayes$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_bayes[, 1],y = coords_bayes[, 2],z = coords_bayes[, 3],
  i = tri_bayes[, 1] - 1,j = tri_bayes[, 2] - 1,k = tri_bayes[, 3] - 1,
  intensity = beta_estimated,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = FALSE,
  name = "Bayesian",
  hoverinfo = "text",
  text = paste("Bayesian:", round(beta_estimated, 6))
)

# Update layout with annotations
fig_comparison <- fig_comparison %>% layout(
  title = "Comparison: True vs Bayesian Coefficients",
  scene = list(
    xaxis = list(title = "X"),
    yaxis = list(title = "Y"),
    zaxis = list(title = "Z"),
    aspectmode = "data",
    camera = list(eye = list(x = 0, y = -2.5, z = 0.3)),
    annotations = list(
      list(x = -offset, y = 0, z = 1.5, text = "TRUE", showarrow = FALSE),
      list(x = offset, y = 0, z = 1.5, text = "BAYESIAN", showarrow = FALSE)
    )
  )
)

print(fig_comparison)
















### coordinate ascent variational inference
sourceCpp("robustLinReg.cpp")

# ### read generated train/test splits
trainIdx <- readRDS("./trainIdx.rds")
testIdx <- readRDS("./testIdx.rds")

# Prepare data (same preprocessing as Gibbs sampler)
y_centered <- y_residualized - mean(y_residualized)
# x_continuous_centered <- x_continuous - mean(x_continuous)
# modelControl <- cbind(rep(1, n_subjects), x_continuous_centered)
imagingFeatures <- final_fit$imagingFeatures
groupID <- cluster_id - 1
n <- nrow(imagingFeatures)

# ### generate train/test splits
# trainIdx <- sample(n, round(9.0*n/10.0), replace = FALSE)
# testIdx <- setdiff(1:n, trainIdx)

# print standard deviations of train/test responses to make sure they
# are roughly equal
print(paste0("train Y standard deviation: ",sd(y_centered[trainIdx])))
print(paste0("test Y standard deviation: ",sd(y_centered[testIdx])))

# ### save generated train/test splits
# saveRDS(trainIdx,"./trainIdx.rds")
# saveRDS(testIdx,"./testIdx.rds")

### save data for python SVI script
write.csv(imagingFeatures[trainIdx,],"./trainImagingFeatures.csv")
write.csv(imagingFeatures[testIdx,],"./testImagingFeatures.csv")
write.csv(y_residualized[trainIdx],"./train_y_residualized.csv")
write.csv(y_residualized[testIdx],"./test_y_residualized.csv")

# Call robustLinReg
# X and testX are lists with a single element containing the imaging features
res_cavi <- robustLinReg(
  X = list(cbind(rep(1,length(trainIdx)),imagingFeatures[trainIdx,])),
  Y = y_centered[trainIdx],
  testX = list(cbind(rep(1,length(testIdx)),imagingFeatures[testIdx,])),
  testY = y_centered[testIdx],
  maxIter = 500,
  numCycles = 1,
  intervalToPrint = 1,
  lambdaStartVec = list(1.0/lambdaSq_true),
  A = 1.0,
  B = list(1.0),
  tol = 1e-4
)

# ### save Bayesian scalar on image regression model
# saveRDS(res_cavi,"./res_cavi.rds")

### read Bayesian scalar on image regression model
res_cavi <- readRDS("./res_cavi.rds")

# View results
cat("\nFinal Results:\n")
cat("Test MSE:", res_cavi$testMSE, "\n")
cat("Test R²:", res_cavi$testR2, "\n")
cat("Estimated regression variance:", res_cavi$evRegVar, "\n")







### visualize CAVI model
### Extract and transform CAVI coefficients
# Get coefficients in basis space (evBeta1 contains the estimated beta)
beta_basis_cavi <- res_cavi$evBeta1

# Transform back to vertex space
beta_estimated_cavi <- t(t(final_fit$Psi) * sqrt(final_fit$lambda)) %*% beta_basis_cavi

# Calculate correlation with true coefficients
correlation_cavi <- cor(beta_estimated_cavi, beta_true)

cat("\nCAVI model performance:\n")
cat("  Correlation (whole sphere):", correlation_cavi, "\n\n")

### Visualize CAVI Coefficients - Whole Sphere
cat("Creating smooth visualization for CAVI Coefficients...\n")
fig_cavi <- visualize_sphere_smooth(
  coords = sphere_coord,
  activation = beta_estimated_cavi,
  title = "CAVI Coefficients - Whole Sphere (Smooth)",
  color_palette = "Plasma"
)

print(fig_cavi)

### Three-way comparison: True vs Ridge vs CAVI with labels
cat("Creating three-way comparison with labels...\n")

offset <- 2.5

# True coefficients (left position)
coords_true <- sphere_coord
coords_true[, 1] <- sphere_coord[, 1] - offset

# Ridge coefficients (middle position)
coords_ridge <- sphere_coord

# CAVI coefficients (right position)
coords_cavi <- sphere_coord
coords_cavi[, 1] <- sphere_coord[, 1] + offset

# Create meshes
fig_comparison <- plot_ly()

# True coefficients
hull_true <- convhulln(coords_true, options = "FA")
tri_true <- hull_true$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_true[, 1],y = coords_true[, 2],z = coords_true[, 3],
  i = tri_true[, 1] - 1,j = tri_true[, 2] - 1,k = tri_true[, 3] - 1,
  intensity = beta_true,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = TRUE,
  colorbar = list(title = "Coefficient", x = 1.15),
  name = "True",
  hoverinfo = "text",
  text = paste("True:", round(beta_true, 6))
)

# Ridge coefficients
hull_ridge <- convhulln(coords_ridge, options = "FA")
tri_ridge <- hull_ridge$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_ridge[, 1],y = coords_ridge[, 2],z = coords_ridge[, 3],
  i = tri_ridge[, 1] - 1,j = tri_ridge[, 2] - 1,k = tri_ridge[, 3] - 1,
  intensity = beta_estimated,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = FALSE,
  name = "Ridge",
  hoverinfo = "text",
  text = paste("Ridge:", round(beta_estimated, 6))
)

# CAVI coefficients
hull_cavi <- convhulln(coords_cavi, options = "FA")
tri_cavi <- hull_cavi$hull

fig_comparison <- fig_comparison %>% add_trace(
  x = coords_cavi[, 1],y = coords_cavi[, 2],z = coords_cavi[, 3],
  i = tri_cavi[, 1] - 1,j = tri_cavi[, 2] - 1,k = tri_cavi[, 3] - 1,
  intensity = beta_estimated_cavi,
  type = "mesh3d",
  colorscale = "Plasma",
  showscale = FALSE,
  name = "CAVI",
  hoverinfo = "text",
  text = paste("CAVI:", round(beta_estimated_cavi, 6))
)

# Add text labels above each sphere
# Position labels at the top (max z coordinate + offset)
label_z <- max(sphere_coord[, 3]) + 0.3

fig_comparison <- fig_comparison %>% add_trace(
  x = c(-offset, 0, offset),
  y = c(0, 0, 0),
  z = c(label_z, label_z, label_z),
  type = "scatter3d",
  mode = "text",
  text = c("TRUE", "RIDGE", "CAVI"),
  textfont = list(size = 16, color = "black", family = "Arial Black"),
  showlegend = FALSE,
  hoverinfo = "none"
)

# Update layout
fig_comparison <- fig_comparison %>% layout(
  title = "Comparison: True vs Ridge vs CAVI Coefficients",
  scene = list(
    xaxis = list(title = "X"),
    yaxis = list(title = "Y"),
    zaxis = list(title = "Z"),
    aspectmode = "data",
    camera = list(eye = list(x = 0, y = -3, z = 0.3))
  )
)

print(fig_comparison)




















