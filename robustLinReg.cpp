#include <RcppArmadillo.h>
#include <math.h>
#include <chrono>
#include <thread>
using namespace Rcpp;


// [[Rcpp::depends(RcppArmadillo)]]


class featureSet {
public:
  arma::Mat<double> features;
  arma::Mat<double> testFeatures;
  arma::Mat<double> covBeta;
  arma::Col<double> evBeta;
  arma::Mat<double> U;
  arma::Col<double> D;
  arma::Mat<double> V;
  arma::Mat<double> XTX;
  arma::Mat<double> tV;
  arma::Mat<double> Z;
  arma::Col<double> Zdiag;
  int numWeights;
  double evRecLambdaScale;
  double evRecLambda;
  double EB;
  double B;
  
  void construct(arma::Mat<double> X, arma::Mat<double> testX,
                 double lambdaStart, double priorVal) {
    features = X;
    testFeatures = testX;
    numWeights = features.n_cols;
    evBeta = arma::zeros(numWeights);
    covBeta = arma::zeros(numWeights, numWeights);
    arma::svd_econ(U,D,V,features);
    tV = arma::trans(V);
    Z = arma::eye(D.n_elem, D.n_elem);
    Zdiag = arma::vec(D.n_elem);
    XTX = trans(features)*features;
    evRecLambda = lambdaStart;
    B = priorVal;
  }
};


// [[Rcpp::export]]
// Group Level Priors Fast Linear Regression
List robustLinReg(List X, NumericVector Y, List testX, NumericVector testY,
                  int maxIter, int numCycles, int intervalToPrint,
                  List lambdaStartVec, double A, List B, double tol) {
  
  featureSet feat[X.size()];
  for (int i = 0; i < X.size(); i++) {
    feat[i].construct(as<arma::mat>(X[i]),as<arma::mat>(testX[i]),lambdaStartVec[i],B[i]);
  }
  
  arma::Col<double> response = as<arma::vec>(Y);
  arma::Col<double> testResponse = as<arma::vec>(testY);
  int n = feat[0].features.n_rows;
  
  // Calculate MSE
  arma::Col<double> testPred;
  arma::Col<double> trainResiduals = arma::vec(n);
  arma::Col<double> testResiduals = arma::vec(n);
  double testR2;
  double trainMSE;
  double testMSE;
  double oldTrainMSE = 1.0;
  
  double evRecRegVar = 1.0;
  double evRecRegVarScale;
  double evRecRegVarNum;
  double evRecRegVarDenom;
  
  for (int iter = 1; iter < maxIter; iter++) {
    // Calculate covariance matrices
    for (int i = 0; i < X.size(); i++) {
      feat[i].Zdiag = pow(evRecRegVar*pow(feat[i].D, 2.0)
                            + evRecRegVar*feat[i].evRecLambda, -1.0);
      feat[i].Z.diag() = feat[i].Zdiag;
      feat[i].covBeta = feat[i].V*feat[i].Z*feat[i].tV;
    }
    
    // Cycle through regression coefficient updates
    for (int i = 0; i < numCycles; i++) {
      for (int j = 0; j < X.size(); j++) {
        trainResiduals = response;
        for (int k = 0; k < X.size(); k++) {
          trainResiduals = trainResiduals - feat[k].features*feat[k].evBeta;
        }
        feat[j].evBeta = evRecRegVar*feat[j].covBeta*trans(feat[j].features)
          *(trainResiduals + feat[j].features*feat[j].evBeta);
      } 
    }
    
    // Update global shrinkage parameters and regression variance
    for (int i = 0; i < X.size(); i++) {
      feat[i].EB = arma::as_scalar(trans(feat[i].evBeta)*feat[i].evBeta
                                     + arma::trace(feat[i].covBeta));
      feat[i].evRecLambdaScale = 1.0/(feat[i].evRecLambda + pow(feat[i].B, -2.0));
      feat[i].evRecLambda = (feat[i].numWeights + 1.0)/arma::as_scalar(2.0*feat[i].evRecLambdaScale
                                                                         + evRecRegVar*feat[i].EB);
    }
    
    trainResiduals = response;
    for (int i = 0; i < X.size(); i++) {
      trainResiduals = trainResiduals - feat[i].features*feat[i].evBeta;
    }
    evRecRegVarScale = 1.0/(evRecRegVar + pow(A, -2.0));
    evRecRegVarNum = n + 1.0;
    evRecRegVarDenom = 2.0*evRecRegVarScale + arma::dot(trainResiduals,trainResiduals);
    for (int i = 0; i < X.size(); i++) {
      evRecRegVarNum = evRecRegVarNum + feat[i].numWeights;
      evRecRegVarDenom = evRecRegVarDenom + feat[i].evRecLambda*feat[i].EB
      + arma::trace(feat[i].XTX*feat[i].covBeta);
    }
    evRecRegVar = evRecRegVarNum/evRecRegVarDenom;
    if (iter % intervalToPrint == 0) {
      testPred = feat[0].testFeatures*feat[0].evBeta;
      for (int i = 1; i < X.size(); i++) {
        testPred = testPred + feat[i].testFeatures*feat[i].evBeta;
      }
      testResiduals = testResponse - testPred;
      trainMSE = mean(square(trainResiduals));
      testMSE = mean(square(testResiduals));
      testR2 = arma::as_scalar(pow(arma::cor(testResponse,testPred), 2.0));
      Rcout << "Iter: " << iter
            << ", Train MSE: " << trainMSE
            << ", Test MSE: " << testMSE
            << ", Rec Lambda Sq: " << feat[0].evRecLambda << "\n";
      if (abs(oldTrainMSE - trainMSE) < tol) {
        break;
      } else {
        oldTrainMSE = trainMSE;
      }
    }
  }
  
  testPred = feat[0].testFeatures*feat[0].evBeta;
  for (int i = 1; i < X.size(); i++) {
    testPred = testPred + feat[i].testFeatures*feat[i].evBeta;
  }
  testResiduals = testResponse - testPred;
  testMSE = mean(square(testResiduals));
  testR2 = arma::as_scalar(pow(arma::cor(testResponse,testPred), 2.0));
  
  double evRegVarNum = 2.0*evRecRegVarScale + arma::dot(trainResiduals,trainResiduals);
  double evRegVarDenom = n;
  for (int i = 0; i < X.size(); i++) {
    evRegVarNum = evRegVarNum + feat[i].evRecLambda*feat[i].EB
    + arma::trace(trans(feat[i].features)*feat[i].features*feat[i].covBeta);
    evRegVarDenom = evRegVarDenom + feat[i].numWeights;
    
  }
  double evRegVar = evRecRegVarNum/evRecRegVarDenom;
  
  return Rcpp::List::create(Named("evRegVar") = evRegVar,
                            Named("testMSE") = testMSE,
                            Named("testR2") = testR2,
                            Named("evBeta1") = feat[0].evBeta);
}
