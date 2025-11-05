








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

# generate spherical coordinates using the fibonacci lattice method
generate_sphere_coords <- function(n_points) {
  golden_ratio <- (1+sqrt(5))/2
  i <- seq(0,n_points-1)
  
  z <- 1-(2*i+1)/n_points
  radius <- sqrt(1-z^2.0)
  phi <- 2*pi*i/golden_ratio
  x <- radius*cos(phi)
  y <- radius*sin(phi)
  coords <- cbind(x,y,z)
  
  return(coords)
}

### generate synthetic brain activation data using Gaussian Process with independent noise
generate_synthetic_activation <- function(gp_coords,n_subjects,a = 0.001,b = 20.0,
                                          noise_sd = 0.1,add_mean_shift = TRUE,
                                          mean_shift = 0.5) {
  
  n_vertices <- nrow(gp_coords)
  norms_sq <- rowSums(gp_coords^2.0)
  
  # vectorized computation
  dist_matrix_sq <- as.matrix(dist(gp_coords,method = "euclidean"))^2.0
  norms_outer_sum <- outer(norms_sq,norms_sq,"+")
  K <- exp(-a*norms_outer_sum - b*dist_matrix_sq)
  K <- K + diag(1e-6,n_vertices)
  L <- chol(K)
  
  # vectorized sampling of brain activations for each subject
  Z <- matrix(rnorm(n_subjects * n_vertices),nrow = n_vertices,ncol = n_subjects)
  noise_matrix <- matrix(rnorm(n_subjects * n_vertices,0,noise_sd),
                         nrow = n_vertices,ncol = n_subjects)
  gp_samples <- t(L) %*% Z
  activation_matrix <- t(gp_samples + noise_matrix)
  
  # add mean shift to covariates to create non-zero mean
  if (add_mean_shift) {
    # add spatially varying mean that is stronger on right hemisphere
    hemisphere_weights <- ifelse(gp_coords[,1] > 0,1.0,0.8)
    for (i in 1:n_subjects) {
      activation_matrix[i,] <- activation_matrix[i,] + mean_shift * hemisphere_weights
    }
  }
  
  return(activation_matrix)
}

### generate regression coefficients using Gaussian Process with lambdaSq scaling
generate_gp_coefficients_scaled <- function(gp_coords,a_coef = 0.01,b_coef = 5.0,
                                            target_sd = 0.0001) {
  
  n_vertices <- nrow(gp_coords)
  norms_sq <- rowSums(gp_coords^2.0)
  
  ### compute expected variance from the kernel to determine lambdaSq
  # given K(x,x') = lambdaSq * exp(-a*(||x||^2 + ||x'||^2) - b*||x-x'||^2)
  # the marginal variance at point x is V[f(x)] = K(x,x) = lambdaSq * exp(-2*a*||x||^2)
  
  # calculate lambdaSq to achieve target SD
  expected_variances_base <- exp(-2*a_coef*norms_sq)
  expected_sd_base <- sqrt(mean(expected_variances_base))
  lambdaSq <- (target_sd/expected_sd_base)^2.0
  
  # vectorized computation of covariance matrix
  dist_matrix_sq <- as.matrix(dist(gp_coords,method = "euclidean"))^2.0
  
  # compute the outer sum of norms_sq
  norms_outer_sum <- outer(norms_sq,norms_sq,"+")
  
  # vectorized covariance matrix computation
  K <- lambdaSq*exp(-a_coef*norms_outer_sum - b_coef*dist_matrix_sq)
  K <- K + diag(1e-9,n_vertices)
  
  # generate sample
  L <- chol(K)
  z <- rnorm(n_vertices)
  gp_sample <- t(L) %*% z
  coefficients <- as.vector(gp_sample)
  
  return(list(coefficients = coefficients,
              lambdaSq = lambdaSq,
              a_coef = a_coef,
              b_coef = b_coef))
}

### generate synthetic brain-like data for whole sphere
n_vertices_total <- 6000

### parameters for data generation
a_data <- 0.001
b_data <- 20.0

# generate coefficients with fixed a=0.001, b=5.0, and lambdaSq scaling
# no mean shift needed for coefficients since covariates now have non-zero mean
target_sd <- 0.001

### read generated data
sphere_coord <- readRDS("./sphere_coord.rds")
cortex_dat <- readRDS("./cortex_dat.rds")
coef_result <- readRDS("./coef_result.rds")

# # generate coordinates for entire sphere (use Cartesian directly)
# sphere_coord <- generate_sphere_coords(n_vertices_total)
# ### Generate covariates and coefficients for entire sphere
# # Add mean shift to covariates to create non-zero imaging signal
# cortex_dat <- generate_synthetic_activation(sphere_coord,n_subjects = 3000,
#                                             a = a_data,b = b_data,add_mean_shift = TRUE,
#                                             mean_shift = 0.5)
# coef_result <- generate_gp_coefficients_scaled(sphere_coord,a_coef = 0.001,b_coef = 5.0,
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

### generate response variable with hierarchical structure
intercept_global <- 0.0
beta_continuous_mean <- 1.0
noise_sd_response <- 1.0
n_subjects <- nrow(cortex_dat)

# create cluster variable with 20 unique IDs
n_clusters <- 20
cluster_id <- rep(1:n_clusters,length.out = n_subjects)
cluster_id <- sample(cluster_id)

# generate continuous covariate
x_continuous <- rnorm(n_subjects,mean = 0,sd = 1)

# generate random effects for each cluster
sd_intercept <- 0.1
sd_slope <- 0.1
random_intercepts <- rnorm(n_clusters,mean = 0,sd = sd_intercept)
random_slopes <- rnorm(n_clusters,mean = 0.5,sd = sd_slope)

# generate response variable
y <- rep(0,n_subjects)

for (i in 1:n_subjects) {
  cluster <- cluster_id[i]
  
  fixed_effects <- intercept_global + beta_continuous_mean * x_continuous[i]
  random_effects <- random_intercepts[cluster] + random_slopes[cluster] * x_continuous[i]
  image_effects <- sum(cortex_dat[i,] * beta_true)
  
  y[i] <- fixed_effects + random_effects + image_effects
}

y <- y + rnorm(n_subjects,0,noise_sd_response)

# residualize response variable
mixed_data <- data.frame(y = y,
                         x_continuous = x_continuous,
                         cluster_id = as.factor(cluster_id))

lmer_model <- lmer(y ~ x_continuous + (1 + x_continuous | cluster_id),
                   data = mixed_data,
                   REML = TRUE)

y_predicted_lmer <- predict(lmer_model)
y_residualized <- y - y_predicted_lmer

#########
### Ridge Regression
#########
### function to fit ridge regression model with fixed basis parameters (3D Cartesian)
fit_ridge_with_basis_3d <- function(a_basis,b_basis,cortex,response,
                                    coords_3d,poly_degree,
                                    alpha_param = 0.0,nfolds = 10) {
  
  # calculate basis functions for 3D Cartesian space
  Psi <- GP.eigen.funcs.fast(coords_3d,poly_degree = poly_degree,a = a_basis,b = b_basis)
  lambda <- GP.eigen.value(poly_degree = poly_degree,a = a_basis,b = b_basis,d = ncol(coords_3d))
  
  # transform to basis representation
  imagingFeatures <- cortex %*% t(t(Psi) * sqrt(lambda))
  
  # fit elastic net regression with cross-validation
  cv_model <- cv.glmnet(x = imagingFeatures,
                        y = response,
                        alpha = alpha_param,
                        nfolds = nfolds,
                        standardize = TRUE,
                        type.measure = "mse")
  
  rmse <- sqrt(min(cv_model$cvm))
  
  return(list(rmse = rmse,
              cv_model = cv_model,
              Psi = Psi,
              lambda = lambda,
              imagingFeatures = imagingFeatures))
}

### use fixed basis parameters
# use same a and b values as coefficient generation
a_basis <- 0.001
b_basis <- 5.0
poly_degree <- 20

# calculate expected number of basis functions
n_basis_expected <- choose(poly_degree + 3,3)

### fit model with fixed parameters
final_fit <- fit_ridge_with_basis_3d(a_basis = a_basis,
                                     b_basis = b_basis,
                                     cortex = cortex_dat,
                                     response = y_residualized,
                                     coords_3d = sphere_coord,
                                     poly_degree = poly_degree)

# get final model
alpha_param <- 0.0
final_model <- glmnet(x = final_fit$imagingFeatures,
                      y = y_residualized,
                      alpha = alpha_param,
                      lambda = final_fit$cv_model$lambda.min,
                      standardize = TRUE)

predictions <- predict(final_model,newx = final_fit$imagingFeatures,
                       s = final_fit$cv_model$lambda.min)

mse <- mean((y_residualized - predictions)^2.0)
rmse <- sqrt(mse)
r_squared <- cor(y_residualized,predictions)^2.0

# get coefficients in basis space
beta_basis <- coef(final_model,s = final_fit$cv_model$lambda.min)
beta_basis_vec <- as.vector(beta_basis[-1])

# transform back to vertex space
beta_ridge <- t(t(final_fit$Psi) * sqrt(final_fit$lambda)) %*% beta_basis_vec
correlation_total <- cor(beta_ridge,beta_true)



#########
### Define Visualization Functions
#########
### function to add a mesh trace to an existing plotly figure
add_mesh_trace <- function(fig,coords,activation,name,
                           show_colorbar = FALSE,colorbar_x = 1.1) {
  
  # compute convex hull
  hull_result <- convhulln(coords,options = "FA")
  triangles <- hull_result$hull
  
  # add trace
  fig <- fig %>% add_trace(x = coords[,1],y = coords[,2],z = coords[,3],
                           i = triangles[,1] - 1,j = triangles[,2] - 1,k = triangles[,3] - 1,
                           intensity = activation,
                           type = "mesh3d",
                           colorscale = "Plasma",
                           showscale = show_colorbar,
                           colorbar = list(title = "Coefficient",x = colorbar_x),
                           name = name,
                           hoverinfo = "text",
                           text = paste(paste0(name,":"),round(activation,6)))
  
  return(fig)
}

### function to create side-by-side comparison of multiple coefficient maps
create_comparison_plot <- function(sphere_coord,coef_list,labels,
                                   title = "Coefficient Comparison",
                                   offset = 2.5,add_text_labels = TRUE) {
  
  # validate inputs
  n_models <- length(coef_list)
  if (length(labels) != n_models) {
    stop("Number of labels must match number of coefficient vectors")
  }
  
  # calculate x-positions for each sphere
  # center the display around x=0
  total_width <- (n_models - 1) * offset
  x_positions <- seq(-total_width/2,total_width/2,length.out = n_models)
  
  # create base figure
  fig <- plot_ly()
  
  # add each mesh
  for (i in 1:n_models) {
    coords_offset <- sphere_coord
    coords_offset[,1] <- sphere_coord[,1] + x_positions[i]
    fig <- add_mesh_trace(fig = fig,
                          coords = coords_offset,
                          activation = coef_list[[i]],
                          name = labels[i],
                          show_colorbar = (i == 1),
                          colorbar_x = 1.15)
  }
  
  # add text labels
  if (add_text_labels) {
    label_z <- max(sphere_coord[,3]) + 0.3
    fig <- fig %>% add_trace(x = x_positions,
                             y = rep(0,n_models),
                             z = rep(label_z,n_models),
                             type = "scatter3d",
                             mode = "text",
                             text = labels,
                             textfont = list(size = 16,color = "black",family = "Arial Black"),
                             showlegend = FALSE,
                             hoverinfo = "none")
  }
  
  # update layout
  fig <- fig %>% layout(title = title,
                        scene = list(xaxis = list(title = "X"),
                                     yaxis = list(title = "Y"),
                                     zaxis = list(title = "Z"),
                                     aspectmode = "data",
                                     camera = list(eye = list(x = 0,y = -2.5,z = 0.3))))
  
  return(fig)
}

# visualize true coefficients
fig_true <- create_comparison_plot(sphere_coord = sphere_coord,
                                   coef_list = list(beta_true),
                                   labels = c("TRUE"),
                                   title = "True Coefficients")
print(fig_true)

# visualize ridge regression coefficients
fig_ridge <- create_comparison_plot(sphere_coord = sphere_coord,
                                    coef_list = list(beta_ridge),
                                    labels = c("RIDGE"),
                                    title = "Ridge Coefficients")
print(fig_ridge)

### true vs ridge comparison
fig_true_ridge <- create_comparison_plot(sphere_coord = sphere_coord,
                                         coef_list = list(beta_true,beta_ridge),
                                         labels = c("TRUE","RIDGE"),
                                         title = "Comparison: True vs Ridge Coefficients")
print(fig_true_ridge)

# save comparison figure
saveRDS(fig_true_ridge,"./fig_true_ridge.rds")








#########
### Fully Bayesian Inference
#########
sourceCpp("robustLinRegMMGS.cpp")

y_centered <- y - mean(y)
x_continuous_centered <- x_continuous - mean(x_continuous)
modelControl <- cbind(rep(1,n_subjects),x_continuous_centered)
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
# saveRDS(res,"./soirModel.rds")

### read Bayesian scalar on image regression model
res <- readRDS("./soirModel.rds")

### process MCMC draws
postControlBeta <- res$controlBeta[,(numDiscard+2):(numEpochs+1)]
postNeuroBeta <- res$neuroBeta[,(numDiscard+2):(numEpochs+1)]
postLambdaSq <- res$lambdaSq[(numDiscard+2):(numEpochs+1)]
postLambdaSqScale <- res$lambdaSqScale[(numDiscard+2):(numEpochs+1)]
postRegVar <- res$regVar[(numDiscard+2):(numEpochs+1)]
postRegVarScale <- res$regVarScale[(numDiscard+2):(numEpochs+1)]
postReVar <- res$reVar[,(numDiscard+2):(numEpochs+1)]

### print train and test MSE
trainMSE = mean((y_centered - res$trainPostPred)^2.0)
print(paste0("Train MSE: ",trainMSE))

trainMSE = mean((y_centered - res$trainPartialPostPred)^2.0)
print(paste0("Train MSE: ",trainMSE))


### MCMC diagnostics
meIdx <- 1
slopeIdx <- 1

trace_df <- data.frame(iter = 1:length(postControlBeta[meIdx,]),
                       intercept = postControlBeta[meIdx,])
ggplot(trace_df,aes(x=iter,y=intercept)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postControlBeta[meIdx,]),color = "red") +
  ggtitle("Intercept Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))

trace_df <- data.frame(iter = 1:length(postNeuroBeta[slopeIdx,]),
                       intercept = postNeuroBeta[slopeIdx,])
ggplot(trace_df,aes(x=iter,y=intercept)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postNeuroBeta[slopeIdx,]),color = "red") +
  ggtitle("Slope Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))

trace_df <- data.frame(iter = 1:length(postLambdaSq),lambdaSq = postLambdaSq)
ggplot(trace_df,aes(x=iter,y=lambdaSq)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postLambdaSq),color = "red") +
  ggtitle("LambdaSq Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))

hist_df <- data.frame(lambdaSq = postLambdaSq)
ggplot(hist_df,aes(x=lambdaSq)) + geom_histogram(bins = 100,color="darkblue",fill="lightblue") +
  ggtitle("LambdaSq Posterior Density") +
  theme(plot.title = element_text(hjust = 0.5))

trace_df <- data.frame(iter = 1:length(postRegVar),regVar = postRegVar)
ggplot(trace_df,aes(x=iter,y=regVar)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postRegVar),color = "red") +
  ggtitle("Reg Var Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))

hist_df <- data.frame(regVar = postRegVar)
ggplot(hist_df,aes(x=regVar)) + geom_histogram(bins = 100,color="darkblue",fill="lightblue") +
  ggtitle("Reg Var Posterior Density") +
  theme(plot.title = element_text(hjust = 0.5))

trace_df <- data.frame(iter = 1:length(postReVar[1,]),reVar = postReVar[1,])
ggplot(trace_df,aes(x=iter,y=reVar)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postReVar[1,]),color = "red") +
  ggtitle("Random Effect Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))

trace_df <- data.frame(iter = 1:length(postReVar[2,]),reVar = postReVar[2,])
ggplot(trace_df,aes(x=iter,y=reVar)) + geom_line(color="darkblue") +
  geom_hline(yintercept=mean(postReVar[2,]),color = "red") +
  ggtitle("Random Effect Trace Plot") +
  theme(plot.title = element_text(hjust = 0.5))

### transform Bayesian coefficients
beta_basis_bayes <- res$neuroBeta
beta_bayes <- colSums(t(final_fit$Psi %*% (diag(sqrt(final_fit$lambda)) %*% postNeuroBeta)))

### visualize coefficients from fully Bayesian inference
fig_bayes <- create_comparison_plot(sphere_coord = sphere_coord,
                                    coef_list = list(beta_bayes),
                                    labels = c("Gibbs"),
                                    title = "Gibbs Coefficients")
print(fig_bayes)

### true vs Bayesian comparison
fig_true_gibbs <- create_comparison_plot(sphere_coord = sphere_coord,
                                         coef_list = list(beta_true,beta_bayes),
                                         labels = c("TRUE","GIBBS"),
                                         title = "Comparison: True vs Bayesian Coefficients")
print(fig_true_gibbs)

# save comparison figure
saveRDS(fig_true_gibbs,"./fig_true_gibbs.rds")





#########
### Coordinate Ascent Variational Inference
#########
sourceCpp("robustLinReg.cpp")

# read generated train/test splits
trainIdx <- readRDS("./trainIdx.rds")
testIdx <- readRDS("./testIdx.rds")

# prepare data (same preprocessing as Gibbs sampler)
y_centered <- y_residualized - mean(y_residualized)
imagingFeatures <- final_fit$imagingFeatures
groupID <- cluster_id - 1
n <- nrow(imagingFeatures)

# print standard deviations of train/test responses to make sure they
# are roughly equal
print(paste0("train Y standard deviation: ",sd(y_centered[trainIdx])))
print(paste0("test Y standard deviation: ",sd(y_centered[testIdx])))

# ### save data for python SVI script
# write.csv(imagingFeatures[trainIdx,],"./trainImagingFeatures.csv")
# write.csv(imagingFeatures[testIdx,],"./testImagingFeatures.csv")
# write.csv(y_residualized[trainIdx],"./train_y_residualized.csv")
# write.csv(y_residualized[testIdx],"./test_y_residualized.csv")

### calculate linear regression model using CAVI
# allows for multiple neuroimaging covariate sets, but in this implementation
# we are only using one set
# does not include an intercept term because the global shrinkage parameter
# has to be applied to the entire design matrix
# res_cavi <- robustLinReg(X = list(imagingFeatures[trainIdx,]),
#                          Y = y_centered[trainIdx],
#                          testX = list(imagingFeatures[testIdx,]),
#                          testY = y_centered[testIdx],
#                          maxIter = 400,
#                          numCycles = 1,
#                          intervalToPrint = 1,
#                          lambdaStartVec = list(1.0/lambdaSq_true),
#                          A = 1.0,
#                          B = list(1.0),
#                          tol = 1e-6)
# 
# saveRDS(res_cavi,"./res_cavi.rds")

### read Bayesian scalar on image regression model
res_cavi <- readRDS("./res_cavi.rds")

### extract and transform CAVI coefficients
beta_basis_cavi <- res_cavi$evBeta1
beta_cavi <- t(t(final_fit$Psi) * sqrt(final_fit$lambda)) %*% matrix(beta_basis_cavi[,],nrow = 1771)
correlation_cavi <- cor(beta_cavi,beta_true)

### visualize CAVI coefficients
fig_cavi <- create_comparison_plot(sphere_coord = sphere_coord,
                                   coef_list = list(beta_cavi),
                                   labels = c("CAVI"),
                                   title = "CAVI Coefficients")
print(fig_cavi)

### true vs Gibbs vs CAVI
fig_true_gibbs_cavi <- create_comparison_plot(sphere_coord = sphere_coord,
                                              coef_list = list(beta_true,beta_bayes,beta_cavi),
                                              labels = c("TRUE","GIBBS","CAVI"),
                                              title = "Comparison: True vs Gibbs vs CAVI Coefficients",
                                              add_text_labels = TRUE)
print(fig_true_gibbs_cavi)

# save comparison figure
saveRDS(fig_true_gibbs_cavi,"./fig_true_gibbs_cavi.rds")









#########
### Stochastic Variational Inference
#########
# load the posterior means computed from Python
posterior_mean_beta_svi <- read.csv("./posterior_mean_svi.csv",header = FALSE)$V1

# map coefficients back to Cartesian space
# need to omit the intercept term in the coefficients from the SVI model
beta_svi <- t(final_fit$Psi %*% (diag(sqrt(final_fit$lambda)) %*% matrix(posterior_mean_beta_svi[2:length(posterior_mean_beta_svi)],
                                                                         nrow = length(posterior_mean_beta_svi) - 1)))

# visualize SVI coefficients
fig_svi <- create_comparison_plot(sphere_coord = sphere_coord,
                                  coef_list = list(beta_svi),
                                  labels = c("SVI"),
                                  title = "SVI Coefficients")
print(fig_svi)

# true vs CAVI vs SVI comparison
fig_true_cavi_svi <- create_comparison_plot(sphere_coord = sphere_coord,
                                            coef_list = list(beta_true,beta_cavi,beta_svi),
                                            labels = c("TRUE","CAVI","SVI"),
                                            title = "True vs CAVI vs SVI")
print(fig_true_cavi_svi)

# save comparison figure
saveRDS(fig_true_cavi_svi,"./fig_true_cavi_svi.rds")















