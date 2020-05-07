#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include "sampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
using std::cout;
using std::endl;


Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}


void Sampler::setEnergies(int MCcycles) {
  for (int i = 0; i < MCcycles; i++){
    m_Energies.push_back(0);
  }

}

void Sampler::setGradients() {
  int nx = m_system->getNumberOfInputs();
  int nh = m_system->getNumberOfHidden();

  m_aDelta.resize(nx);
  m_EaDelta.resize(nx);
  m_agrad.resize(nx);

  m_bDelta.resize(nh);
  m_EbDelta.resize(nh);
  m_bgrad.resize(nh);

  m_wDelta.resize(nx*nh);
  m_EwDelta.resize(nx*nh);
  m_wgrad.resize(nx*nh);

}


void Sampler::sample(bool acceptedStep, int MCcycles) {
    // Making sure the sampling variable(s) are initialized at the first step.
    if (m_stepNumber == 0) {
        m_cumulativeEnergy = 0;
        m_cumulativeEnergy2 = 0;
        m_DeltaPsi = 0;
        m_DerivativePsiE = 0;
        setGradients();
    }


    double localEnergy = m_system->getHamiltonian()->
                     computeLocalEnergy(m_system->getNetwork());

    Eigen::VectorXd temp_aDelta = m_system->getNetwork()->computeBiasAgradients();
    Eigen::VectorXd temp_bDelta = m_system->getNetwork()->computeBiasBgradients();
    Eigen::VectorXd temp_wDelta = m_system->getNetwork()->computeWeightsgradients();



    m_cumulativeEnergy  += localEnergy;
    m_cumulativeEnergy2  += localEnergy*localEnergy;

    m_aDelta += temp_aDelta;
    m_bDelta += temp_bDelta;
    m_wDelta += temp_wDelta;

    m_EaDelta += temp_aDelta*localEnergy;
    m_EbDelta += temp_bDelta*localEnergy;
    m_EwDelta += temp_wDelta*localEnergy;

    m_stepNumber++;
}

void Sampler::printOutputToTerminal(double total_time, double acceptedStep) {

    // Initialisers
    int     np = m_system->getNumberOfParticles();
    int     nd = m_system->getNumberOfDimensions();
    int     ms = m_system->getNumberOfMetropolisSteps();
    int     p  = m_system->getWaveFunction()->getNumberOfParameters();
    double  ef = m_system->getEquilibrationFraction();
    std::vector<double> pa = m_system->getWaveFunction()->getParameters();

    if (m_system->getPrintToTerminal()){
      cout << endl;
      cout << "  -- System info -- " << endl;
      cout << " Number of particles  : " << np << endl;
      cout << " Number of dimensions : " << nd << endl;
      cout << " Number of Monte Carlo cycles : 10^" << std::log10(ms) << endl;
      cout << " Number of equilibration steps  : 10^" << std::log10(std::round(ms*ef)) << endl;
      cout << endl;
      cout << "  -- Wave function parameters -- " << endl;
      cout << " Number of parameters : " << p << endl;
      for (int i=0; i < p; i++) {
          cout << " Parameter " << i+1 << " : " << pa.at(i) << endl;
      }
      cout << endl;
      cout << "  -- Results -- " << endl;
      cout << " Energy : " << m_energy << endl;
      cout << " Variance : " << m_variance << endl;
      cout << " Time : " << total_time << endl;
      cout << " # Accepted Steps : " << acceptedStep << endl;
      cout << endl;
  }
}

void Sampler::computeAverages(double total_time, double acceptedStep) {
    /* Compute the averages of the sampled quantities.
     */
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    int N = m_system->getNumberOfParticles(); // Number of Particles
    int MCcycles = m_system->getNumberOfMetropolisSteps(); // Number of Monte Carlo steps
    double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles

    m_energy = m_cumulativeEnergy *norm;
    m_cumulativeEnergy2 = m_cumulativeEnergy2 *norm;
    m_cumulativeEnergy = m_cumulativeEnergy *norm;
    m_variance = (m_cumulativeEnergy2 - m_cumulativeEnergy*m_cumulativeEnergy)*norm;
    m_STD = sqrt(m_variance);

    m_aDelta /= MCcycles;
    m_bDelta /= MCcycles;
    m_wDelta /= MCcycles;

    m_EaDelta /= MCcycles;
    m_EbDelta /= MCcycles;
    m_EwDelta /= MCcycles;

    // Compute gradients
    m_agrad = 2*(m_EaDelta - m_cumulativeEnergy*m_aDelta);
    m_bgrad = 2*(m_EbDelta - m_cumulativeEnergy*m_bDelta);
    m_wgrad = 2*(m_EwDelta - m_cumulativeEnergy*m_wDelta);

    // Update weights and biases
    m_system->getNetwork()->optimizeWeights(m_agrad, m_bgrad, m_wgrad);

    m_totalTime = total_time;
    m_acceptedStep = acceptedStep;
}



void Sampler::WriteResultstoFile(ofstream& ofile, int MCcycles){
  int N = m_system->getNumberOfParticles(); // Number of Particles
  double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles

  double Energy = m_cumulativeEnergy * norm;
  double CumulativeEnergy2 = m_cumulativeEnergy2 *norm;
  double CumulativeEnergy = m_cumulativeEnergy *norm;
  double Variance = CumulativeEnergy2 - CumulativeEnergy*CumulativeEnergy;
  double STD = sqrt(Variance*norm);


  //ofile << "\n";
  //ofile << setw(15) << setprecision(8) << MCcycles; // # Monte Carlo cycles (sweeps per lattice)
  ofile << setw(15) << setprecision(8) << Energy << endl; // Mean energy
  //ofile << setw(15) << setprecision(8) << m_cumulativeEnergy << endl; // Variance
  //ofile << setw(15) << setprecision(8) << STD; // # Standard deviation

}
