#include "system.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <omp.h>
#include <time.h>
#include "sampler.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "Hamiltonians/hamiltonian.h"
#include "InitialStates/initialstate.h"
#include "Math/random.h"
#include <armadillo>
using namespace arma;


bool System::metropolisStep(int i) {
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

     double a = RandomNumberGenerator(gen) - 0.5; // Random number
     double b = RandomNumberGenerator(gen) - 0.5; // Random number
     double c = RandomNumberGenerator(gen) - 0.5; // Random number

     int Dim = getNumberOfDimensions(); // The Dimensions

     double r, wfold, wfnew;

     // Initial Position
     wfold = getWaveFunction()->evaluate(m_particles);


     // Trial position moving one particle at the time in all dimensions

     m_particles[i]->adjustPosition(m_stepLength*a, 0);
     if (Dim>1){
       m_particles[i]->adjustPosition(m_stepLength*b, 1);
       if (Dim>2){
         m_particles[i]->adjustPosition(m_stepLength*c, 2);
       }
     }

     wfnew = getWaveFunction()->evaluate(m_particles);

     // Metropolis test
	if ( RandomNumberGenerator(gen) <= wfnew*wfnew/(wfold*wfold) ){

    return true;
  }

  else{

     m_particles[i]->adjustPosition(-m_stepLength*a, 0);
     if (Dim>1){
       m_particles[i]->adjustPosition(-m_stepLength*b, 1);
       if (Dim>2){
         m_particles[i]->adjustPosition(-m_stepLength*c, 2);
       }
     }
    return false;
  }

}


bool System::ImportanceMetropolisStep(int i) {
    /* Perform the actual Metropolis step: Choose a particle at random and
     * change it's position by a random amount, and check if the step is
     * accepted by the Metropolis test (compare the wave function evaluated
     * at this new position with the one at the old position).
     */
     // Initialize the seed and call the Mersienne algo
     std::random_device rd;
     std::mt19937_64 gen(rd());
     // Set up the uniform distribution for x \in [[0, 1]
     std::normal_distribution<double> Normal(0.0,1.0);
     std::uniform_real_distribution<double> Uniform(0.0,1.0);


     double a = Normal(gen); // Random number
     double b = Normal(gen); // Random number
     double c = Normal(gen); // Random number

     int Dim = getNumberOfDimensions(); // The Dimensions

     double r, wfold, wfnew, poschange;
     std::vector<double> posold, posnew, qfold, qfnew;

     // Initial Position
     posold = getParticles()[i]->getPosition();
     wfold = getWaveFunction()->evaluate(m_particles);
     qfold = getHamiltonian()->computeQuantumForce(m_particles, i);


     // Trial position moving one particle at the time in all dimensions
     poschange = a*sqrt(m_timeStep) + qfold[0]*m_timeStep*m_diffusionCoefficient;
     m_particles[i]->adjustPosition(poschange, 0);
     if (Dim > 1){
       poschange = b*sqrt(m_timeStep) + qfold[1]*m_timeStep*m_diffusionCoefficient;
       m_particles[i]->adjustPosition(poschange, 1);
       if (Dim > 2){
         poschange = c*sqrt(m_timeStep) + qfold[2]*m_timeStep*m_diffusionCoefficient;
         m_particles[i]->adjustPosition(poschange, 2);
       }
     }


     posnew = getParticles()[i]->getPosition();
     wfnew = getWaveFunction()->evaluate(m_particles);
     qfnew = getHamiltonian()->computeQuantumForce(m_particles, i);

     // Greens function
     double greensFunction = 0;
     for (int k = 0; k < Dim; k++)
     {
       greensFunction += 0.5*(qfold[k] + qfnew[k])*(m_diffusionCoefficient*m_timeStep*0.5*(qfold[k] - qfnew[k]) - posnew[k] + posold[k]);
     }
     greensFunction = exp(greensFunction);

     // #Metropolis-Hastings test to see whether we accept the move
	if ( Uniform(gen) <= greensFunction*wfnew*wfnew/(wfold*wfold) ){

    return true;
  }

  else{

    poschange = a*sqrt(m_timeStep) + qfold[0]*m_timeStep*m_diffusionCoefficient;
    m_particles[i]->adjustPosition(-poschange, 0);
    if (Dim > 1){
      poschange = b*sqrt(m_timeStep) + qfold[1]*m_timeStep*m_diffusionCoefficient;
      m_particles[i]->adjustPosition(-poschange, 1);
      if (Dim > 2){
        poschange = c*sqrt(m_timeStep) + qfold[2]*m_timeStep*m_diffusionCoefficient;
        m_particles[i]->adjustPosition(-poschange, 2);
      }
    }
    return false;
  }

}

void System::runMetropolisSteps(ofstream& ofile, int numberOfMetropolisSteps) {
    m_particles                 = m_initialState->getParticles();
    m_sampler                   = new Sampler(this);
    m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
    m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);

    int N = getNumberOfParticles();
    double counter = 0;
    bool acceptedStep;

    double start_time = omp_get_wtime();

    for (int i = 1; i <= numberOfMetropolisSteps; i++) {
        // Trial position moving one particle at the time
        for (int j = 0; j < N; j++){
          if (getImportanceSampling()){
              acceptedStep = ImportanceMetropolisStep(j);
          }
          else{
              acceptedStep = metropolisStep(j);
          }

          /* Here you should sample the energy (and maybe other things using
          * the m_sampler instance of the Sampler class. Make sure, though,
          * to only begin sampling after you have let the system equilibrate
          * for a while. You may handle this using the fraction of steps which
          * are equilibration steps; m_equilibrationFraction.
          */

        }
        //if (i >= m_equilibrationFraction*m_numberOfMetropolisSteps){
        m_sampler->sample(acceptedStep, i);
        if (getOneBodyDensity() != true){
            m_sampler->WriteResultstoFile(ofile, i);
        }

        //}
        counter += acceptedStep;
    }
    m_sampler->computeAverages();
    m_sampler->printOutputToTerminal();
    cout << "# Accepted Step = " << counter/numberOfMetropolisSteps << endl;
    if (getOneBodyDensity()){
      m_sampler->WriteOneBodyDensitytoFile(ofile);
    }
  double end_time = omp_get_wtime();
  double total_time = end_time - start_time;
  m_sampler->setTime(total_time);
  cout << "Time is \n " << total_time << endl;
}

void System::setNumberOfParticles(int numberOfParticles) {
    m_numberOfParticles = numberOfParticles;
}

void System::setNumberOfDimensions(int numberOfDimensions) {
    m_numberOfDimensions = numberOfDimensions;
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
}
void System::setBinCounter(int new_count, int index){
  m_binCounter[index] = new_count;
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
}

bool System::setImportanceSampling(bool importance_sampling){
  m_importance_sampling = importance_sampling;
}

bool System::setNumericalDerivative(bool numerical_derivative){
  m_numerical_dericative = numerical_derivative;
}
