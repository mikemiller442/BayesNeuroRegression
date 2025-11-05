#include <RcppArmadillo.h>
#include <math.h>
#include <chrono>
#include <thread>
using namespace Rcpp;


// [[Rcpp::depends(RcppArmadillo)]]


// Object that contains the data and parameters associated with the fixed
// and random effects associated with a Bayesian mixed effects model
class FeatureSetGS {
public:
  arma::Mat<double> features,testFeatures,invPriorMat;
  arma::Mat<double> beta;
  arma::Mat<double> XTX;
  arma::Col<double> XTy,Y;
  arma::Col<double> postBeta;
  int numWeights,numEpochs;
  double interceptPrior,betaPrior;
  
  // Constructor method
  // Z is a matrix of mixed effects data in the training set.
  // testZ is a matrix of mixed effects data in the testing set.
  // resp contains the response variable in the training set.
  // ne corresponds to the number of epochs used in training.
  void construct(arma::Mat<double> Z,arma::Mat<double> testZ,
                 arma::Col<double> resp,int ne) {
    features = Z;
    testFeatures = testZ;
    Y = resp;
    XTX = arma::trans(features)*features;
    XTy = arma::trans(features)*Y;
    numWeights = features.n_cols;
    numEpochs = ne;
    beta = arma::zeros(numWeights,numEpochs + 1);
    beta.col(0) = arma::randn(numWeights,arma::distr_param(0.0,0.0001));
    interceptPrior = 100.0;
    betaPrior = 1.0;
  }
};


// Object that contains the data and parameters associated with a single task
// contrast in a regression model. Unlike the CAVI method this model is
// restricted to a single neuroimaging feature set
class FeatureSetHPGS {
public:
  arma::Mat<double> features,testFeatures;
  arma::Mat<double> beta;
  arma::Col<double> lambdaSqScale,lambdaSq;
  arma::Col<double> evBeta,postBeta;
  int numWeights;
  int numEpochs;
  arma::Mat<double> XTX;
  arma::Col<double> XTy;
  arma::Col<double> Y;
  arma::Mat<double> U;
  arma::Col<double> D;
  arma::Mat<double> V;
  double lambdaSqScaleAlpha,lambdaSqScaleBeta;
  double lambdaSqAlpha,lambdaSqBeta,lambdaSqPrior;
  
  // Constructor method
  // X is a matrix of neuroimaging data in the training set.
  // testX is a matrix of neuroimaging data in the testing set.
  // resp contains the response variable in the training set.
  // lss is the starting value for the lambdaSq parameter in Gibbs sampling.
  // ne corresponds to the number of epochs used in training.
  void construct(arma::Mat<double> X,arma::Mat<double> testX,
                 arma::Col<double> resp,double lss,int ne) {
    features = X;
    testFeatures = testX;
    Y = resp;
    XTX = arma::trans(features)*features;
    XTy = arma::trans(features)*Y;
    numWeights = features.n_cols;
    numEpochs = ne;
    arma::svd(U,D,V,features);
    beta = arma::zeros(numWeights,numEpochs + 1);
    beta.col(0) = arma::randn(numWeights,arma::distr_param(0.0,0.00001));
    lambdaSqPrior = 1.0;
    lambdaSqScale = arma::zeros(numEpochs + 1);
    lambdaSqScale(0) = 1.0;
    lambdaSq = arma::zeros(numEpochs + 1);
    lambdaSq(0) = lss;
  }
};


// Object that contains the data necessary to estimate a Bayesian mixed
// effects model with a single neuroimaging feature set
class LinRegModelGS {
public:
  arma::Col<double> Y,testY,regVarScale,regVar,XTy;
  arma::Mat<double> features,XTX;
  int n,testN,numEpochs;
  double regVarScaleAlpha,regVarScaleBeta,regVarAlpha,regVarBeta,regVarPrior;
  arma::Mat<double>* Sigma;
  arma::Mat<double> SigmaScale;
  arma::Mat<double> mixX;
  arma::Mat<double> testMixX;
  arma::Mat<double> R;
  arma::Mat<double> testR;
  arma::Mat<double> C;
  arma::Mat<double> testC;
  arma::Col<double> group;
  arma::Col<double> testGroup;
  arma::Col<double> trainPostPred,trainPartialPostPred;
  arma::Col<double> testPostPred,testPartialPostPred;
  arma::Col<double> trainResiduals;
  arma::Mat<double> LS;
  arma::Mat<double> reVar;
  arma::Col<double> wishScaleAlpha,wishScaleBeta;
  double wishA;
  arma::Mat<double> wishB,invSigma;
  int numGroups,numFE,numRE;
  arma::Col<double> SSP;
  double nu;
  FeatureSetGS me;
  FeatureSetHPGS hp;
  
  // Constructor method
  // Z/testZ is a matrix of the control variables in the training/testing data sets.
  // X/testX is a matrix of the neuroimaging in the training/testing data sets.
  // resp/testResp is the outcome variable in the training/testing data sets.
  // groupVec/testGroupVec is the vector of group indicator variables in the
  // training/testing data sets.
  // mixedX/testMixedX are the matrices containing the random effects data
  // in the training/testing data sets.
  // lss is the starting value for the lambdaSq parameter in Gibbs sampling.
  // ss is the starting value for the diagonal values of the random effects
  // prior covariance matrix.
  // ng is the number of groups
  // ne is the number of epochs for the Gibbs sampling
  void construct(arma::Mat<double> Z,arma::Mat<double> testZ,
                 arma::Mat<double> X,arma::Mat<double> testX,
                 arma::Col<double> resp,arma::Col<double> testResp,
                 arma::Col<double> groupVec,arma::Col<double> testGroupVec,
                 arma::Mat<double> mixedX,arma::Mat<double> testMixedX,
                 double lss,double ss,int ng,int ne) {
    
    // model attributes
    Y = resp;
    testY = testResp;
    n = Y.n_elem;
    testN = testY.n_elem;
    numEpochs = ne;
    regVarPrior = 1.0;
    regVarScale = arma::zeros(numEpochs + 1);
    regVarScale(0) = 1.0;
    regVar = arma::zeros(numEpochs + 1);
    regVar(0) = 1.0;
    
    // mixed effects attributes
    mixX = mixedX;
    testMixX = testMixedX;
    group = groupVec;
    testGroup = testGroupVec;
    numGroups = ng;
    numFE = Z.n_cols;
    numRE = mixX.n_cols;
    SSP = arma::ones(numRE,1);
    SigmaScale = arma::zeros(numRE,numEpochs + 1);
    SigmaScale.col(0) = arma::ones(numRE,1);
    Sigma = new arma::Mat<double>[numEpochs + 1];
    Sigma[0] = arma::eye(numRE,numRE)*ss;
    nu = 2.0;
    
    // construct mixed effects design matrix
    R = arma::zeros(n,numRE*numGroups);
    testR = arma::zeros(testN,numRE*numGroups);
    for (int i = 0; i < n; i++) {
      R.submat(i,numRE*group[i],i,numRE*(group[i]+1)-1) = mixX.row(i);
    }
    for (int i = 0; i < testN; i++) {
      testR.submat(i,numRE*testGroup[i],i,numRE*(testGroup[i]+1)-1) = testMixX.row(i);
    }
    C = arma::join_rows(Z,R);
    testC = arma::join_rows(testZ,testR);
    me.construct(C,testC,Y,ne);
    hp.construct(X,testX,Y,lss,ne);
    features = arma::join_rows(me.features,hp.features);
    XTX = arma::trans(features)*features;
    XTy = arma::trans(features)*Y;
  }
  
  //
  void processRE(int numDiscard) {
    reVar = arma::zeros(numRE,numEpochs + 1);
    for (int i = 0; i < numEpochs + 1; i++) {
      reVar.col(i) = Sigma[i].diag();
    }
  }
  
  // calculate regression coefficient posterior means
  void postMeans(int numDiscard) {
    me.postBeta = (1.0/(numEpochs - numDiscard + 1))
      *arma::sum(me.beta.submat(0,numDiscard,me.numWeights-1,numEpochs),1);
    hp.postBeta = (1.0/(numEpochs - numDiscard + 1))
      *arma::sum(hp.beta.submat(0,numDiscard,hp.numWeights-1,numEpochs),1);
  }
  
  // calculate posterior predictions
  void postPred() {
    trainPostPred = me.features*me.postBeta + hp.features*hp.postBeta;
    testPostPred = me.testFeatures*me.postBeta + hp.testFeatures*hp.postBeta;
    trainPartialPostPred = me.features*me.postBeta;
    testPartialPostPred = me.testFeatures*me.postBeta;
  }
  
  void sampleVarParams(int epoch) {
    
    // sample scale parameters for random effect covariance matrix
    wishScaleAlpha = arma::ones(numRE,1)*(nu + numRE)/2.0;
    invSigma = arma::inv_sympd(Sigma[epoch]);
    wishScaleBeta = nu*invSigma.diag() + arma::pow(SSP,-2.0);
    for (int i = 0; i < numRE; i++) {
      SigmaScale(i,epoch+1) = 1.0/arma::randg(arma::distr_param(wishScaleAlpha(i),
                                              1.0/wishScaleBeta(i)));
    }
    
    // sample random effect covariance matrix
    wishA = nu + numRE + numGroups - 1.0;
    wishB = 2.0*nu*arma::diagmat(arma::pow(SigmaScale.col(epoch+1),-1.0));
    for (int i = 0; i < numGroups; i++) {
      wishB = wishB + me.beta.submat(numFE+i*numRE,epoch+1,numFE+(i+1)*numRE-1,epoch+1)
        *arma::trans(me.beta.submat(numFE+i*numRE,epoch+1,numFE+(i+1)*numRE-1,epoch+1));
    }
    Sigma[epoch+1] = arma::iwishrnd(wishB,wishA);
    
    // sample global shrinkage parameter
    hp.lambdaSqScaleAlpha = 1.0;
    hp.lambdaSqScaleBeta = 1.0/hp.lambdaSq(epoch) + pow(hp.lambdaSqPrior,-2.0);
    hp.lambdaSqScale(epoch + 1) = 1.0/arma::randg(arma::distr_param(hp.lambdaSqScaleAlpha,
                                                  1.0/hp.lambdaSqScaleBeta));
    hp.lambdaSqAlpha = 0.5*(hp.numWeights + 1.0);
    hp.lambdaSqBeta = 0.5*(1.0/regVar(epoch))*arma::dot(hp.beta.col(epoch + 1),
                           hp.beta.col(epoch + 1)) + 1.0/hp.lambdaSqScale(epoch + 1);
    hp.lambdaSq(epoch + 1) = 1.0/randg(arma::distr_param(hp.lambdaSqAlpha,
                                       1.0/hp.lambdaSqBeta));
    
    // sample regression variance
    trainResiduals = Y - me.features*me.beta.col(epoch + 1) - hp.features*hp.beta.col(epoch + 1);
    regVarScaleAlpha = 1.0;
    regVarScaleBeta = 1.0/regVar(epoch) + pow(regVarPrior,-2.0);
    regVarScale(epoch + 1) = 1.0/arma::randg(arma::distr_param(regVarScaleAlpha,
                                             1.0/regVarScaleBeta));
    regVarAlpha = 0.5*(n + numFE + hp.numWeights + 1.0);
    regVarBeta = 1.0/regVarScale(epoch+1) + 0.5*arma::dot(trainResiduals,trainResiduals)
      + 0.5*(1.0/me.interceptPrior)*pow(me.beta(0,epoch+1),2.0)
      + 0.5*(1.0/me.betaPrior)*arma::dot(me.beta.submat(1,epoch+1,numFE-1,epoch+1),
                                         me.beta.submat(1,epoch+1,numFE-1,epoch+1))
      + 0.5*(1.0/hp.lambdaSq(epoch+1))*arma::dot(hp.beta.col(epoch+1),hp.beta.col(epoch+1));
    regVar(epoch + 1) = 1.0/arma::randg(arma::distr_param(regVarAlpha,1.0/regVarBeta));
  }
};


// Estimates a Bayesian mixed effects linear regression model.
// Z/testZ is a matrix of the control variables in the training/testing data sets.
// X/testX is a matrix of the neuroimaging in the training/testing data sets.
// Y/testY is the outcome variable in the training/testing data sets.
// groupVec/testGroupVec is the vector of group indicator variables in the
// training/testing data sets.
// mixedX/testMixedX are the matrices containing the random effects data.
// lambdaSqStart is the starting value for lambdaSq.
// SigmaStart is the starting value for the random effects covariance matrix
// numGroups is the number of groups
// numEpochs is the number of epochs for Gibbs sampling.
// numDiscard is the number of epochs to be discarded as burn-in.
// paramPeriod is the number of epochs before printing updates to console.
// [[Rcpp::export]]
Rcpp::List linRegMMGS(NumericMatrix Z,NumericMatrix testZ,
                      NumericMatrix X,NumericMatrix testX,
                      NumericVector Y,NumericVector testY,
                      NumericVector groupVec,NumericVector testGroupVec,
                      NumericMatrix mixedX,NumericMatrix testMixedX,
                      double lambdaSqStart,double SigmaStart,
                      int numGroups,int numEpochs,int numDiscard,
                      int paramPeriod) {
  
  // initialize mixed effects model with neuroimaging covariates
  LinRegModelGS mod;
  mod.construct(as<arma::mat>(Z),as<arma::mat>(testZ),
                as<arma::mat>(X),as<arma::mat>(testX),
                as<arma::vec>(Y),as<arma::vec>(testY),
                as<arma::vec>(groupVec),as<arma::vec>(testGroupVec),
                as<arma::mat>(mixedX),as<arma::mat>(testMixedX),
                lambdaSqStart,SigmaStart,numGroups,numEpochs);
  
  arma::Col<double> trainResiduals,a,evBetaBar,stdNormal,mu,beta;
  arma::Mat<double> covMat,A;
  arma::Col<double> wishScaleAlpha,wishScaleBeta;
  arma::Mat<double> wishB,invSigma;
  
  // initialize parameters using Gibbs sampling iteration
  for (int epoch = 0; epoch < numEpochs; epoch++) {
    
    // display current state of Markov chain
    if (epoch % paramPeriod == 0) {
      Rcout << "epoch: " << epoch << "\n";
      Rcout << "regVar: " << mod.regVar(epoch) << "\n";
      Rcout << "lambdaSq: " << mod.hp.lambdaSq(epoch) << "\n";
      Rcout << "Sigma: " << mod.Sigma[epoch](0,0) << "\n";
      Rcout << "Sigma: " << mod.Sigma[epoch](1,0) << "\n";
      Rcout << "Sigma: " << mod.Sigma[epoch](1,1) << "\n";
    }
    
    // calculate conditional distribution of regression coefficients
    A = (1.0/mod.regVar(epoch))*mod.XTX;
    mod.me.invPriorMat = arma::zeros(mod.me.numWeights,mod.me.numWeights);
    mod.me.invPriorMat(0,0) = (1.0/mod.regVar(epoch))*(1.0/mod.me.interceptPrior);
    mod.me.invPriorMat.submat(1,1,mod.numFE-1,mod.numFE-1) = arma::eye(mod.numFE-1,mod.numFE-1)
      *(1.0/mod.regVar(epoch))*(1.0/mod.me.betaPrior);
    mod.me.invPriorMat.submat(mod.numFE,mod.numFE,mod.numFE + mod.numGroups*mod.numRE-1,
                              mod.numFE + mod.numGroups*mod.numRE-1) = arma::kron(arma::eye(mod.numGroups,mod.numGroups),
                                                                                  arma::inv(mod.Sigma[epoch]));
    A.submat(0,0,mod.me.numWeights-1,mod.me.numWeights-1) = A.submat(0,0,mod.me.numWeights-1,mod.me.numWeights-1)
      + mod.me.invPriorMat;
    A.submat(mod.me.numWeights,mod.me.numWeights,mod.me.numWeights+mod.hp.numWeights-1,
             mod.me.numWeights+mod.hp.numWeights-1) = A.submat(mod.me.numWeights,mod.me.numWeights,
                                                               mod.me.numWeights+mod.hp.numWeights-1,
                                                               mod.me.numWeights+mod.hp.numWeights-1)
      + arma::eye(mod.hp.numWeights,mod.hp.numWeights)*(1.0/mod.regVar(epoch))
                                                      *(1.0/mod.hp.lambdaSq(epoch));
    
    // jointly sample the regression coefficients
    mod.LS = arma::chol(A,"lower");                                                
    mu = arma::solve(arma::trimatu(arma::trans(mod.LS)),arma::solve(arma::trimatl(mod.LS),mod.XTy));
    stdNormal = arma::randn(mod.me.numWeights+mod.hp.numWeights,arma::distr_param(0.0,1.0));
    beta = mu + arma::solve(arma::trimatu(arma::trans(mod.LS)),stdNormal);
    
    // update the regression coefficients for the mixed effect and neuroimaging
    // feature sets respectively
    mod.me.beta.col(epoch + 1) = beta.subvec(0,mod.me.numWeights-1);
    mod.hp.beta.col(epoch + 1) = beta.subvec(mod.me.numWeights,mod.me.numWeights+mod.hp.numWeights-1);
    
    // sample variance parameters
    mod.sampleVarParams(epoch);
  }
  
  mod.processRE(numDiscard);
  mod.postMeans(numDiscard);
  mod.postPred();
  
  return Rcpp::List::create(Named("postControlBeta") = mod.me.postBeta,
                            Named("postNeuroBeta") = mod.hp.postBeta,
                            Named("controlBeta") = mod.me.beta,
                            Named("neuroBeta") = mod.hp.beta,
                            Named("lambdaSqScale") = mod.hp.lambdaSqScale,
                            Named("lambdaSq") = mod.hp.lambdaSq,
                            Named("regVarScale") = mod.regVarScale,
                            Named("regVar") = mod.regVar,
                            Named("reVar") = mod.reVar,
                            Named("trainPostPred") = mod.trainPostPred,
                            Named("testPostPred") = mod.testPostPred,
                            Named("trainPartialPostPred") = mod.trainPartialPostPred,
                            Named("testPartialPostPred") = mod.testPartialPostPred);
}
