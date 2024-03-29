#include "AdHocIntegrator.hpp"

#include <cstdlib>
#include <string>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_min.h>

#include "eval/ArrayReservoir.hpp"
#include "eval/ArrayRestorer.hpp"
#include "math/Arithmetics.hpp"
#include "eval/FullCachePolicy.hpp"
#include "BranchLengthOptimizer.hpp"


#include "common.h"
#include "comm/Communicator.hpp"
#include "lcfit.h"
#include "lcfit_select.h"
#include "lcfit2.h"


struct log_likelihood_data {
  BranchPlain branch;
  TreeAln* traln;
  AbstractParameter* param;
  LikelihoodEvaluator* eval;
  size_t n_evals;
};


double log_likelihood_callback(double t, void* data)
{
  log_likelihood_data *lnl_data = static_cast<log_likelihood_data*>(data);

  BranchPlain branch = lnl_data->branch;
  TreeAln& traln = *(lnl_data->traln);
  AbstractParameter* param = lnl_data->param;
  LikelihoodEvaluator& eval = *(lnl_data->eval);

  auto b = traln.getBranch(branch, param);
  b.setConvertedInternalLength(traln, param, t);
  traln.setBranch(b, param);
  eval.evaluate(traln, branch, false);

  ++(lnl_data->n_evals);
  return traln.getTrHandle().likelihood;
}

double log_likelihood_d1f(double t, void* data)
{
    gsl_function F;
    F.function = &log_likelihood_callback;
    F.params = data;

    double result = 0.0;
    double abserr = 0.0;

    gsl_deriv_central(&F, t, 1e-6, &result, &abserr);

    return result;
}

double log_likelihood_d2f(double t, void* data)
{
    gsl_function F;
    F.function = &log_likelihood_d1f;
    F.params = data;

    double result = 0.0;
    double abserr = 0.0;

    gsl_deriv_central(&F, t, 1e-5, &result, &abserr);

    return result;
}

double inv_log_likelihood_callback(double t, void* data)
{
    return -log_likelihood_callback(t, data);
}

double log_likelihood_t0(double guess, double min_t, double max_t, void* data)
{
    gsl_min_fminimizer* s;
    gsl_function F;

    F.function = &inv_log_likelihood_callback;
    F.params = data;

    s = gsl_min_fminimizer_alloc(gsl_min_fminimizer_goldensection);
    gsl_min_fminimizer_set(s, &F, guess, min_t, max_t);

    const int MAX_ITER = 100;
    int iter = 0;
    int status;

    do {
        gsl_min_fminimizer_iterate(s);

        guess = gsl_min_fminimizer_x_minimum(s);
        min_t = gsl_min_fminimizer_x_lower(s);
        max_t = gsl_min_fminimizer_x_upper(s);

        status = gsl_min_test_interval(min_t, max_t, 0.0, 1e-5);
        ++iter;
    } while (status == GSL_CONTINUE && iter < MAX_ITER);

    if (iter == MAX_ITER) {
        fprintf(stderr, "WARNING: maximum number of iterations reached during minimization\n");
    }

    gsl_min_fminimizer_free(s);

    return guess;
}

AdHocIntegrator::AdHocIntegrator(TreeAln &traln, std::shared_ptr<TreeAln> debugTree, randCtr_t seed, ParallelSetup* pl)
{
  auto && plcy = std::unique_ptr<ArrayPolicy>(new FullCachePolicy(traln, true, true));
  auto&& res = std::make_shared<ArrayReservoir>(false);
  auto eval = LikelihoodEvaluator(traln, plcy.get() , res, pl);
  
#ifdef DEBUG_LNL_VERIFY
  eval.setDebugTraln(debugTree);
#endif
  
  // s.t. we do not have to care about the branch length linking problem 
  assert(traln.getNumberOfPartitions() == 1 ); 

  auto params = std::vector<std::unique_ptr<AbstractParameter> > {}; 
  params.emplace_back(std::unique_ptr<AbstractParameter>(new BranchLengthsParameter(0,0, {0}))); 
  for(nat i = 0; i < traln.getNumberOfPartitions(); ++i)
    params[0]->addPartition(i);

  double lambda = 10;

  params[0]->setPrior(std::unique_ptr<AbstractPrior>(new ExponentialPrior(lambda)));

  auto proposals = std::vector<std::unique_ptr<AbstractProposal> > {};   
  proposals.emplace_back( new BranchIntegrator (ProposalRegistry::initBranchLengthMultiplier)); 
  proposals[0]->addPrimaryParameter( std::move(params[0])); 

  std::vector<ProposalSet> pSets; 

  integrationChain = make_unique<Chain>( seed, traln, proposals, pSets, std::move(eval), false );
}


void AdHocIntegrator::copyTree(const TreeAln &traln)
{
  auto &myTree = integrationChain->getTralnHandle();
  myTree = traln; 
}


void AdHocIntegrator::prepareForBranch( const BranchPlain &branch,  const TreeAln &otherTree)
{
  copyTree(otherTree); 
  auto &traln = integrationChain->getTralnHandle();

  auto ps = integrationChain->getProposalView(); 
  auto paramView = ps[0]->getBranchLengthsParameterView();
  assert(ps.size() == 1 );   
  auto integrator = dynamic_cast<BranchIntegrator*>(ps[0]); 
  integrator->setToPropose(branch);      

  // std::cout << "important TODO: has the branch been reset? will not do that here" << std::endl; 

  auto &eval = integrationChain->getEvaluator(); 
  eval.evaluate(traln, branch, true); 
  integrationChain->reinitPrior(); 
}


std::vector<AbstractParameter*> AdHocIntegrator::getBlParamView() const
{
  auto ps = integrationChain->getProposalView(); 
  auto paramView = ps[0]->getBranchLengthsParameterView();
  assert(ps.size() == 1 );   
  return paramView; 
}


std::vector<double> AdHocIntegrator::integrate( const BranchPlain &branch, const TreeAln &otherTree)
{
  auto result =  std::vector<double>{}; 
  auto& traln = integrationChain->getTralnHandle(); 

  // tout << "integrating " << branch << std::endl; 

  prepareForBranch(branch, otherTree); 
  auto paramView = integrationChain->getProposalView()[0]->getBranchLengthsParameterView();
  assert(paramView.size( )== 1 ); 

  auto backup = otherTree.getBranch(branch, paramView[0]); 

  bool converged = false; 
  while(not converged)
    {
      for(nat i = 0; i < 10000; ++i) 
	{	  
	  integrationChain->step();
	  auto elem = traln.getBranch(branch, paramView[0]); 
	  auto iLen = elem.getInterpretedLength(traln, paramView[0]);
	  if (i % 10 == 0)
	    result.push_back(iLen); ; 
	}
      
      auto ess = Arithmetics::getEffectiveSamplingSize(result); 
      converged = ess > 10000.; 
    }

  traln.setBranch(backup, paramView[0]); 
  return result;   
}


void run_lcfit2(std::string runid,
                log_like_function_t lnl_fn, const double tolerance,
                const double min_t, const double max_t,
                const double t0, const double d1, const double d2)
{
  const log_likelihood_data* lnl_data = static_cast<log_likelihood_data*>(lnl_fn.args);

  std::stringstream ss;
  ss << "." << runid << "."
     << lnl_data->branch.getPrimNode() << "-" << lnl_data->branch.getSecNode()
     << ".tab";

  const std::string file_suffix = ss.str();

  // Write out log-likelihoods evaluated at extreme points.
  std::ofstream extremesOut("extremes" + file_suffix);

  const double middle_t = (min_t + max_t) / 2.0;

  extremesOut << min_t << "\t"
              << lnl_fn.fn(min_t, lnl_fn.args) << "\t"
              << t0 << "\t"
              << lnl_fn.fn(t0, lnl_fn.args) << "\t"
              << middle_t << "\t"
              << lnl_fn.fn(middle_t, lnl_fn.args) << "\t"
              << max_t << "\t"
              << lnl_fn.fn(max_t, lnl_fn.args) << std::endl;
}


/** 
    @brief gets the optimimum 
 */ 
double AdHocIntegrator::printOptimizationProcess(const BranchLength& branch, std::string runid, nat nrSteps, Communicator& comm)
{
  auto &traln = integrationChain->getTralnHandle(); 

  auto paramView = integrationChain->getProposalView()[0]->getBranchLengthsParameterView();
  assert(paramView.size() == 1 ); 
  auto tmpBranch = branch; 
  auto &&ss = std::stringstream{} ; 
  ss << "nr-length." << runid  << "." << branch.getPrimNode() << "-" << branch.getSecNode() << ".tab"; 
  auto &&thisOut =  std::ofstream(ss.str()); 

  double result = 0;  
  double curVal = 0.1; 
  tmpBranch.setConvertedInternalLength(traln, paramView[0], curVal); 
  double secDerivative = 0; 
  double firstDerivative = 0; 

  double prevVal = curVal; 
  curVal = tmpBranch.getLength();

  for(nat i = 0; i < nrSteps; ++i )
    {
      auto blo = BranchLengthOptimizer(traln, branch.toPlain(), 1, comm, paramView);
      blo.optimizeBranches(traln); 

      auto optParams = blo.getOptimizedParameters(); 

      tmpBranch.setLength(optParams[0].getOptimum()); 
      firstDerivative  = optParams[0].getFirstDerivative(); 
      secDerivative = optParams[0].getSecondDerivative();

      thisOut << prevVal <<  "\t" << firstDerivative << "\t" << secDerivative << endl; 	
      
      prevVal = tmpBranch.getInterpretedLength(traln, paramView[0]); 

      if(not BoundsChecker::checkBranch(tmpBranch))
	BoundsChecker::correctBranch(tmpBranch); 
      
      traln.setBranch(tmpBranch, paramView[0]); 
      curVal = tmpBranch.getInterpretedLength(traln, paramView[0]); 
    } 

  //
  // Here there be lcfit2.
  //

  const double tolerance = 1e-3;

  auto param = paramView[0];
  auto& eval = integrationChain->getEvaluator();

  // Use formula from LengthPart<double>::getInterpretedLength and
  // internal length min and max from BoundsChecker.
  const double frac_c = traln.getMeanSubstitutionRate(param->getPartitions());
  const double min_t = -log(BoundsChecker::zMax) * frac_c;
  const double max_t = -log(BoundsChecker::zMin) * frac_c;

  log_likelihood_data lnl_data = {branch.toPlain(), &traln, param, &eval, 0};
  log_like_function_t lnl_fn = {&log_likelihood_callback, static_cast<void*>(&lnl_data)};

  if (abs(firstDerivative) < 0.1) {
      // Calculate the maximum-likelihood branch length with GSL.
      const double t0_guess = prevVal;
      const double gsl_t0 = log_likelihood_t0(t0_guess, min_t, max_t, lnl_fn.args);

      // Recalculate the derivatives of the log-likelihood curve with GSL.
      const double gsl_d1_gsl_t0 = log_likelihood_d1f(gsl_t0, lnl_fn.args);
      const double gsl_d2_gsl_t0 = log_likelihood_d2f(gsl_t0, lnl_fn.args);

      fprintf(stderr, "gsl_t0 = %g, gsl_d1(gsl_t0) = %g, gsl_d2(gsl_t0) = %g\n",
              gsl_t0, gsl_d1_gsl_t0, gsl_d2_gsl_t0);

      double lmax_d1;
      double lmax_d2;
      double lmax_t0 = lcfit_maximize(&log_likelihood_callback, &lnl_data,
                                      min_t, max_t, &lmax_d1, &lmax_d2);

      fprintf(stderr, "lmax_t0 = %g, lmax_d1(lmax_t0) = %g, lmax_d2(lmax_t0) = %g\n",
              lmax_t0, lmax_d1, lmax_d2);

      // Override ExaBayes's calculations.
      //prevVal = gsl_t0;
      //firstDerivative = gsl_d1_gsl_t0;
      //secDerivative = gsl_d2_gsl_t0;

      prevVal = lmax_t0;
      firstDerivative = lmax_d1;
      secDerivative = lmax_d2;
  }

  run_lcfit2(runid, lnl_fn, tolerance, min_t, max_t,
             prevVal, firstDerivative, secDerivative);

  //
  // Here ends lcfit2.
  //

  return prevVal; 
}

std::vector<double> tokenize_as_doubles(std::string str, char delim)
{
    std::vector<double> values;

    size_t head = 0;
    size_t tail;

    do {
        tail = str.find(delim, head);
        if (tail == std::string::npos) {
            tail = str.size();
        }

        std::string token = str.substr(head, tail);

        values.push_back(std::stod(token));

        head = tail + 1;
    } while (tail != str.size());

    return values;
}

std::vector<double> get_starting_points(double min_t, double max_t)
{
    char* cstr = getenv("LCFIT_POINTS");

    if (cstr == nullptr) {
        tout << "using default starting points" << endl;
        return std::vector<double>{0.1, 0.15, 0.5, 1.0};
    }

    std::vector<double> points = tokenize_as_doubles(cstr, ',');

    tout << "using " << cstr << " as starting points" << endl;
    return points;
}

bsm_t get_starting_model()
{
    char* cstr = getenv("LCFIT_MODEL");

    if (cstr == nullptr) {
        tout << "using default starting model" << endl;
        return DEFAULT_INIT;
    }

    std::vector<double> values = tokenize_as_doubles(cstr, ',');
    bsm_t m;

    m.c = values[0];
    m.m = values[1];
    m.r = values[2];
    m.b = values[3];

    tout << "using " << cstr << " as starting model" << endl;
    return m;
}

void run_lcfit(std::string runid,
               log_likelihood_data lnl_data, log_like_function_t lnl_fn,
               const double tolerance, const double min_t, const double max_t)
{
  bsm_t model = get_starting_model();
  std::vector<double> ts = get_starting_points(min_t, max_t);
  bool success = true;

  //double ml_t = estimate_ml_t(&lnl_fn, ts.data(), ts.size(), tolerance, &model, &success, min_t, max_t);
  double ml_t = lcfit_fit_auto(lnl_fn.fn, lnl_fn.args, &model, min_t, max_t);

  double d1;
  double d2;
  lcfit_maximize(lnl_fn.fn, lnl_fn.args, min_t, max_t, &d1, &d2);
  
  std::stringstream ss;
  ss << "lcfit." << runid << "."
     << lnl_data.branch.getPrimNode() << "-" << lnl_data.branch.getSecNode()
     << ".tab";
  
  std::ofstream lcfitOut(ss.str());

  lcfitOut << tolerance << "\t"
           << lnl_data.n_evals << "\t"
           << (success ? "true" : "false") << "\t"
           << setprecision(std::numeric_limits<double>::digits10)
           << model.c << "\t"
           << model.m << "\t"
           << model.r << "\t"
           << model.b << "\t"
           << ml_t << "\t"
           << d1 << "\t"
           << d2 << endl;

}

void AdHocIntegrator::createLnlCurve(BranchPlain branch, std::string runid, TreeAln & traln , double minHere, double maxHere, nat numSteps)
{
  auto paramView  = integrationChain->getProposalView()[0]->getBranchLengthsParameterView();
  assert(paramView.size( )== 1 ); 
  auto param = paramView[0]; 
  std::stringstream ss; 
  ss << "lnl." << runid << "." << branch.getPrimNode() << "-" << branch.getSecNode() << ".tab"; 
  std::ofstream thisOut(ss.str());

  auto &eval = integrationChain->getEvaluator(); 
  eval.evaluate(traln, branch, true);

  if(maxHere != minHere)
    {
      for(double i = minHere; i < maxHere+0.00000001 ; i+= (maxHere-minHere)/ numSteps)
	{
	  auto b = traln.getBranch(branch, param); 
	  b.setConvertedInternalLength(traln, paramView[0], i); 
	  traln.setBranch(b, paramView[0]); 
	  eval.evaluate(traln, branch, false);
	  double lnl = traln.getTrHandle().likelihood; 
	  thisOut << i << "\t" << setprecision(std::numeric_limits<double>::digits10) << lnl << endl; 
	}
    }
  else
    thisOut << minHere << "\t" << "NA" << endl; 

  //
  // Here there be lcfit.
  //

  const double tolerance = 1e-3;

  // Use formula from LengthPart<double>::getInterpretedLength and
  // internal length min and max from BoundsChecker.
  const double frac_c = traln.getMeanSubstitutionRate(param->getPartitions());
  const double min_t = -log(BoundsChecker::zMax) * frac_c;
  const double max_t = -log(BoundsChecker::zMin) * frac_c;

  log_likelihood_data lnl_data = {branch, &traln, param, &eval, 0};
  log_like_function_t lnl_fn = {&log_likelihood_callback, static_cast<void*>(&lnl_data)};

  run_lcfit(runid, lnl_data, lnl_fn, tolerance, min_t, max_t);
}


double AdHocIntegrator::getParsimonyLength(TreeAln &traln, const BranchPlain &b, Communicator& comm )
{
  auto pEval =  ParsimonyEvaluator{}; 
  auto branchLength = std::vector<nat>{} ; 
  auto state2pars = pEval.evaluate(traln, b.findNodePtr(traln), true  );
  
  // state2pars.data() = comm.allReduce(state2pars.data()); 

  auto tmp = std::vector<nat>(begin(state2pars), end(state2pars)) ; 
  tmp = comm.allReduce(tmp); 
  
  for(nat i = 0; i < tmp.size() ; ++i)
    state2pars[i] = tmp[i]; 
  
  assert(std::get<0>(state2pars) != 0 || std::get<1>(state2pars) != 0 ); 

  assert(traln.getNumberOfPartitions() == 1 ); 
  auto& partition =  traln.getPartition(0);
  auto length = partition.getUpper() - partition.getLower(); 

  return  double(state2pars[0] ) / double(length) ; 
}

