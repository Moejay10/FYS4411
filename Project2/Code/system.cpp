#include <cassert>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <random>
#include <time.h>
#include <omp.h>
#include "system.h"
#include "sampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "NeuralNetworks/network.h"
#include "Math/random.h"
#include <mpi.h>


bool System::metropolisStep() {
  /* Perform the actual Metropolis step: Choose a particle at random and
   * change it's position by a random amount, and check if the step is
   * accepted by the Metropolis test (compare the wave function evaluated
   * at this new position with the one at the old position).
   */
   // Initialize the seed and call the Mersienne algo
   std::random_device rd;
   std::mt19937_64 gen(rd());
   // Set up the uniform distribution for x \in [[0, 1]
   std::uniform_real_distribution<double> RandomNumberGenerator(0.0,1.0);

   // Set up the uniform distribution for x \in [[0, N]
   std::uniform_int_distribution<int> Inputs(0, getNumberOfParticles()-1);

   int input = Inputs(gen);
   double a = RandomNumberGenerator(gen) - 0.5; // Random number
   double b = RandomNumberGenerator(gen) - 0.5; // Random number
   double c = RandomNumberGenerator(gen) - 0.5; // Random number

   int Dim = getNumberOfDimensions(); // The Dimensions

   double wfold, wfnew;

   // Initial Position
   wfold = getWaveFunction()->evaluate();

   // Trial position moving one particle at the time in all dimensions
   getNetwork()->adjustPositions(m_stepLength*a, 0, input);
   if (Dim>1){
     getNetwork()->adjustPositions(m_stepLength*b, 1, input);
     if (Dim>2){
       getNetwork()->adjustPositions(m_stepLength*c, 2, input);
     }
   }

   wfnew = getWaveFunction()->evaluate();

   // Metropolis test
   if ( RandomNumberGenerator(gen) <= wfnew*wfnew/(wfold*wfold) ){
      return true;
   }

   // return to previous value if Metropolis test is false
   else{
     getNetwork()->adjustPositions(-m_stepLength*a, 0, input);
     if (Dim>1){
       getNetwork()->adjustPositions(-m_stepLength*b, 1, input);
       if (Dim>2){
         getNetwork()->adjustPositions(-m_stepLength*c, 2, input);
       }
     }
      return false;
    }
}


bool System::ImportanceMetropolisStep() {
     // Perform the Importance sampling metropolis step.

     // Initialize the seed and call the Mersienne algo
     std::random_device rd;
     std::mt19937_64 gen(rd());
     // Set up the uniform distribution for x \in [[0, 1]
     std::normal_distribution<double> Normal(0.0,1.0);
     std::uniform_real_distribution<double> Uniform(0.0,1.0);
     // Set up the uniform distribution for x \in [[0, N]
     std::uniform_int_distribution<int> Inputs(0,getNumberOfParticles()-1);

     int input = Inputs(gen);
     double a = Normal(gen); // Random number
     double b = Normal(gen); // Random number
     double c = Normal(gen); // Random number

     int Dim = getNumberOfDimensions(); // The Dimensions
     int N   = getNumberOfParticles(); // The Particles


     double wfold, wfnew, poschange;

     // Initial Position
     vec posold = getNetwork()->getPositions();
     wfold = getWaveFunction()->evaluate();
     vec qfold = 2*(getWaveFunction()->computeFirstDerivative());



     // Trial position moving one particle at the time in all dimensions
     poschange = a*sqrt(m_timeStep) + qfold(input*N + 0)*m_timeStep*m_diffusionCoefficient;
     getNetwork()->adjustPositions(poschange, 0, input);
     if (Dim > 1){
       poschange = b*sqrt(m_timeStep) + qfold(input*N + 1)*m_timeStep*m_diffusionCoefficient;
       getNetwork()->adjustPositions(poschange, 1, input);
       if (Dim > 2){
         poschange = c*sqrt(m_timeStep) + qfold(input*N + 2)*m_timeStep*m_diffusionCoefficient;
         getNetwork()->adjustPositions(poschange, 2, input);
       }
     }

     // evaluate new position
     vec posnew = getNetwork()->getPositions();
     wfnew = getWaveFunction()->evaluate();
     vec qfnew = 2*(getWaveFunction()->computeFirstDerivative());

     // Greens function
     double greensFunction = 0;
     for (int k = 0; k < Dim; k++)
     {
       greensFunction += 0.5*(qfold(input*N + k) + qfnew(input*N + k))*(m_diffusionCoefficient*m_timeStep*0.5*(qfold(input*N + k) - qfnew(input*N + k)) - posnew(input*N + k) + posold(input*N + k));
     }
     greensFunction = exp(greensFunction);

     // #Metropolis-Hastings test to see whether we accept the move
	if ( Uniform(gen) <= greensFunction*wfnew*wfnew/(wfold*wfold) ){

    return true;
  }

  // return to previous value if Metropolis test is false
  else{
    poschange = a*sqrt(m_timeStep) + qfold(input*N + 0)*m_timeStep*m_diffusionCoefficient;
    getNetwork()->adjustPositions(-poschange, 0, input);
    if (Dim > 1){
      poschange = b*sqrt(m_timeStep) + qfold(input*N + 1)*m_timeStep*m_diffusionCoefficient;
      getNetwork()->adjustPositions(-poschange, 1, input);
      if (Dim > 2){
        poschange = c*sqrt(m_timeStep) + qfold(input*N + 2)*m_timeStep*m_diffusionCoefficient;
        getNetwork()->adjustPositions(-poschange, 2, input);
        }
      }

    return false;
  }

}

void System::Gibbs() {
  // Set new hidden variables given positions, according to the logistic sigmoid function
  // (implemented by comparing the sigmoid probability to a uniform random variable)

  // Initialize the seed and call the Mersienne algo
  std::random_device rd;
  std::mt19937_64 gen(rd());
  // Set up the uniform distribution for x \in [[0, 1]
  std::normal_distribution<double> Normal(0.0,1.0);
  std::uniform_real_distribution<double> Uniform(0.0,1.0);
  // Set up the uniform distribution for x \in [[0, N]
  std::uniform_int_distribution<int> Inputs(0,getNumberOfParticles()-1);

  int nx = getNumberOfInputs();
  int nh = getNumberOfHidden();

  double sigma = getWaveFunction()->getParameters()[0];
  double sigma2 = sigma*sigma;

  vec x = getNetwork()->getPositions();
  vec a = getNetwork()->getBiasA();
  vec b = getNetwork()->getBiasB();
  mat w = getNetwork()->getWeigths();

  vec h(nh);
  vec new_positions(nx);

  double z, logisticSigmoid;
  for (int j = 0; j < nh; j++) {
    z = b(j) + (dot(x, w.col(j)))/(sigma2);
    logisticSigmoid = 1.0/(1+exp(-z));
    h(j) = Uniform(gen) < logisticSigmoid;
  }

  // Set new positions (visibles) given hidden, according to normal distribution
  std::normal_distribution<double> distributionX;
  double xMean;
  for (int i = 0; i < nx; i++) {
      xMean = a(i) + dot(w.row(i), h);
      distributionX = std::normal_distribution<double>(xMean, sigma);
      new_positions(i) = distributionX(gen);
  }

  getNetwork()->setPositions(new_positions);
}




void System::runOptimizer(ofstream& ofile, int OptCycles, int numberOfMetropolisSteps) {
  int numberOfProcesses, myRank;
  MPI_Comm_rank (MPI_COMM_WORLD, &myRank);
  MPI_Comm_size (MPI_COMM_WORLD, &numberOfProcesses);

  m_sampler                   = new Sampler(this);
  m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
  m_sampler->setNumberOfMetropolisSteps(m_numberOfMetropolisSteps);

  int localMetropolisSteps = (numberOfMetropolisSteps+numberOfMetropolisSteps*getEquilibrationFraction() )/numberOfProcesses;
  
  double start_time, end_time, total_time;

  m_sampler->setEnergies(m_numberOfMetropolisSteps);

  for (int i = 0; i < OptCycles; i++){
    start_time = omp_get_wtime();

    runMetropolisSteps(ofile, localMetropolisSteps, numberOfProcesses, myRank);

    end_time = omp_get_wtime();
    total_time = end_time - start_time;

    m_sampler->computeAverages(total_time, numberOfProcesses, myRank);
    if (myRank==0){
       m_sampler->printOutputToTerminal(total_time);
       ofile << setw(15) << setprecision(8) << i+1; // Iteration
       ofile << setw(15) << setprecision(8) << m_sampler->getEnergy() << endl; // Mean energy
    }
  }
  //m_sampler->WriteBlockingtoFile(ofile, m_numberOfMetropolisSteps, myRank);  
}


void System::runMetropolisSteps(ofstream& ofile, int localMetropolisSteps, int numberOfProcesses, int myRank) {
  
  double effectiveSamplings = 0;
  double counter = 0;
  bool acceptedStep;
  int localEquilibrationFraction = getEquilibrationFraction()*m_numberOfMetropolisSteps;

  m_sampler->initializeVariables();
  for (int i = 1; i <= localMetropolisSteps; i++) { //to run correct number of MC

    // Choose importance samling or brute force
    if (getImportanceSampling()){
        acceptedStep = ImportanceMetropolisStep();
    }

    else if (getGibbsSampling()){
      Gibbs();
    }

    else{
        acceptedStep = metropolisStep();
    }

    if (i>localEquilibrationFraction){
      effectiveSamplings++; 
      counter += acceptedStep;

      m_sampler->sample();

      m_sampler->Analysis(effectiveSamplings, numberOfProcesses, myRank);
      
    }


  }
  m_sampler->setMCcyles(effectiveSamplings);
  m_sampler->setacceptedSteps(counter);

}

// A lot of setters.
void System::setNumberOfParticles(int numberOfParticles) {
    m_numberOfParticles = numberOfParticles;
}

void System::setNumberOfDimensions(int numberOfDimensions) {
    m_numberOfDimensions = numberOfDimensions;
}

void System::setNumberOfHidden(int numberOfHidden) {
    m_numberOfHidden = numberOfHidden;
}

void System::setStepLength(double stepLength) {
    assert(stepLength >= 0);
    m_stepLength = stepLength;
}

void System::setTimeStep(double timeStep) {
    assert(timeStep >= 0);
    m_timeStep = timeStep;
}

void System::setStepSize(double stepSize) {
    assert(stepSize >= 0);
    m_stepSize = stepSize;
}

void System::setEquilibrationFraction(double equilibrationFraction) {
    assert(equilibrationFraction >= 0);
    m_equilibrationFraction = equilibrationFraction;
}

void System::setHamiltonian(Hamiltonian* hamiltonian) {
    m_hamiltonian = hamiltonian;
}
void System::setNetwork(Network* network) {
    m_network = network;
}

void System::setWaveFunction(WaveFunction* waveFunction) {
    m_waveFunction = waveFunction;
}

void System::setBinVector(double binStartpoint, double binEndpoint, int numberofBins){
  std::vector<double> binVector;
  std::vector<int> binCounter;

  double step = (binEndpoint-binStartpoint)/(numberofBins);
  for (int i = 0; i < numberofBins; i++){
    binVector.push_back((double)i * step);
    binCounter.push_back(0);
  }
  m_binVector = binVector;
  m_binCounter = binCounter;
  m_partclesPerBin = binCounter;
}

void System::setBinCounter(int new_count, int index){
  m_binCounter[index] = new_count;
}

void System::setParticlesPerBin(int index){
  m_binCounter[index]++;
}

void System::setOneBodyDensity(bool oneBodyDensity){
  m_oneBodyDensity = oneBodyDensity;
}

void System::setInitialState(InitialState* initialState) {
    m_initialState = initialState;
}


void System::setDiffusionCoefficient(double diffusionCoefficient) {
    m_diffusionCoefficient = diffusionCoefficient;
}

void System::setBinEndpoint(double binEndpoint) {
    m_binEndpoint = binEndpoint;
}

void System::setBinStartpoint(double binStartpoint) {
    m_binStartpoint = binStartpoint;
}

void System::setNumberofBins(int numberofBins) {
    m_numberofBins = numberofBins;
}

bool System::setRepulsivePotential(bool statement){
  m_statement = statement;
  return m_statement;
}

bool System::setImportanceSampling(bool importance_sampling){
  m_importance_sampling = importance_sampling;
  return m_importance_sampling;
}

bool System::setGibbsSampling(bool gibbs_sampling){
  m_gibbs_sampling = gibbs_sampling;
  return m_gibbs_sampling;
}

bool System::setOptimizer(bool optimizer){
  m_optimizer = optimizer;
  return m_optimizer;
}

bool System::setPrintOutToTerminal(bool print_terminal){
  m_print_terminal = print_terminal;
  return m_print_terminal;
}
