#include "system.h"
#include <cassert>
#include <random>
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

     double a = RandomNumberGenerator(gen); // Random number

     int Dim = getNumberOfDimensions(); // The Dimensions

     double r, wfold, wfnew;

     // Initial Position
     wfold = getWaveFunction()->evaluate(m_particles);


     // Trial position moving one particle at the time in all dimensions
     for (int k = 0; k < Dim; k++)
     {
        m_particles[i]->adjustPosition(a, k);
     }
     wfnew = getWaveFunction()->evaluate(m_particles);

     // Metropolis test
	if ( RandomNumberGenerator(gen) <= wfnew*wfnew/(wfold*wfold) ){

    return true;
  }

  else{

    for (int k = 0; k < Dim; k++){
        m_particles[i]->adjustPosition(-a, k);
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
     std::uniform_real_distribution<double> RandomNumberGenerator(0.0,1.0);

     double a = RandomNumberGenerator(gen); // Random number

     int Dim = getNumberOfDimensions(); // The Dimensions

     double r, wfold, wfnew, poschange;
     std::vector<double> posold, posnew, qfold, qfnew;

     // Initial Position
     posold = getParticles()[i]->getPosition();
     wfold = getWaveFunction()->evaluate(m_particles);
     qfold = getHamiltonian()->computeQuantumForce(m_particles, i);


     // Trial position moving one particle at the time in all dimensions
     for (int k = 0; k < Dim; k++)
     {
        poschange = a*sqrt(m_stepLength) + qfold[k]*m_stepLength*m_diffusionCoefficient;
        m_particles[i]->adjustPosition(poschange, k);
     }
     posnew = getParticles()[i]->getPosition();
     wfnew = getWaveFunction()->evaluate(m_particles);
     qfnew = getHamiltonian()->computeQuantumForce(m_particles, i);

     // Greens function
     double greensFunction = 0;
     for (int k = 0; k < Dim; k++)
     {
       greensFunction += 0.25*(qfold[k] + qfnew[k])*(m_diffusionCoefficient*m_stepLength*(qfold[k] - qfnew[k]) - posnew[k] + posold[k]);
     }
     greensFunction = exp(greensFunction);
     // Metropolis test
	if ( RandomNumberGenerator(gen) <= greensFunction*wfnew*wfnew/(wfold*wfold) ){

    return true;
  }

  else{

    for (int k = 0; k < Dim; k++){
      poschange = a*sqrt(m_stepLength) + qfold[k]*m_stepLength*m_diffusionCoefficient;
      m_particles[i]->adjustPosition(-poschange, k);
     }

    return false;
  }

}

void System::runMetropolisSteps(int numberOfMetropolisSteps) {
    m_particles                 = m_initialState->getParticles();
    m_sampler                   = new Sampler(this);
    m_numberOfMetropolisSteps   = numberOfMetropolisSteps;
    m_sampler->setNumberOfMetropolisSteps(numberOfMetropolisSteps);

    int N = getNumberOfParticles();
    bool acceptedStep;
    for (int i=0; i < numberOfMetropolisSteps; i++) {

        // Trial position moving one particle at the time
        for (int j = 0; j < N; j++){
        acceptedStep = metropolisStep(j);

        //acceptedStep = ImportanceMetropolisStep(j);

        /* Here you should sample the energy (and maybe other things using
         * the m_sampler instance of the Sampler class. Make sure, though,
         * to only begin sampling after you have let the system equilibrate
         * for a while. You may handle this using the fraction of steps which
         * are equilibration steps; m_equilibrationFraction.
         */
         m_sampler->sample(acceptedStep);
      }
    }
    m_sampler->computeAverages();
    m_sampler->printOutputToTerminal();
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

void System::setInitialState(InitialState* initialState) {
    m_initialState = initialState;
}


void System::setDiffusionCoefficient(double diffusionCoefficient) {
    m_diffusionCoefficient = diffusionCoefficient;
}
