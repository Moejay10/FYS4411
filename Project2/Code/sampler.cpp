#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <armadillo>
#include "sampler.h"
#include "system.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"
#include "NeuralNetworks/network.h"
#include <mpi.h>

using std::cout;
using std::endl;
using namespace arma;

Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}

void Sampler::setacceptedSteps(int counter) {
    m_localacceptedSteps = counter;
}


void Sampler::setEnergies(int MCcycles) {
  m_Energies.zeros(MCcycles);
  m_Times.zeros(MCcycles);
}

void Sampler::setBlocking(int MCcycle) {
  m_Blocking.zeros(MCcycle);
}

void Sampler::initializeVariables() {
  int nx = m_system->getNumberOfInputs();
  int nh = m_system->getNumberOfHidden();

  m_globalaDelta.zeros(nx);
  m_globalEaDelta.zeros(nx);

  m_localaDelta.zeros(nx);
  m_localEaDelta.zeros(nx);
  m_agrad.zeros(nx);

  m_globalbDelta.zeros(nh);
  m_globalEbDelta.zeros(nh);
  m_localbDelta.zeros(nh);
  m_localEbDelta.zeros(nh);
  m_bgrad.zeros(nh);

  m_localwDelta.zeros(nx*nh);
  m_localEwDelta.zeros(nx*nh);
  m_globalwDelta.zeros(nx*nh);
  m_globalEwDelta.zeros(nx*nh);
  m_wgrad.zeros(nx*nh);

  m_localcumulativeEnergy = 0;
  m_globalcumulativeEnergy = 0;
  m_localcumulativeEnergy2 = 0;
  m_globalcumulativeEnergy2 = 0;
  m_globalacceptedSteps = 0;
  m_localacceptedSteps = 0;
  m_DeltaPsi = 0;
  m_DerivativePsiE = 0;
  //m_energy=0;
}


void Sampler::sample() {

    vec x = m_system->getNetwork()->getPositions();
    vec b = m_system->getNetwork()->getBiasB();
    mat W = m_system->getNetwork()->getWeigths();
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    vec Q = b + (1.0/sigma2)*(x.t()*W).t();

    double localEnergy = m_system->getHamiltonian()->
                     computeLocalEnergy(m_system->getNetwork(), Q);

    vec temp_aDelta = m_system->getNetwork()->computeBiasAgradients();
    vec temp_bDelta = m_system->getNetwork()->computeBiasBgradients(Q);
    vec temp_wDelta = m_system->getNetwork()->computeWeightsgradients(Q);

    m_localcumulativeEnergy  += localEnergy;
    m_localcumulativeEnergy2  += localEnergy*localEnergy;

    m_localaDelta += temp_aDelta;
    m_localbDelta += temp_bDelta;
    m_localwDelta += temp_wDelta;

    m_localEaDelta += temp_aDelta*localEnergy;
    m_localEbDelta += temp_bDelta*localEnergy;
    m_localEwDelta += temp_wDelta*localEnergy;

    m_stepNumber++;
}

void Sampler::printOutputToTerminal(double total_time) {

    // Initialisers
    int     nx = m_system->getNumberOfInputs();
    int     nh = m_system->getNumberOfHidden();
    int     np = m_system->getNumberOfParticles();
    int     nd = m_system->getNumberOfDimensions();
    int     ms = m_system->getNumberOfMetropolisSteps();
    int     p  = 3;
    double  ef = m_system->getEquilibrationFraction();
    vec pam(3); pam(0) = nx; pam(1) = nh; pam(2) = nx*nh;

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
          cout << " Parameter " << i+1 << " : " << pam(i) << endl;
      }
      cout << endl;
      cout << "  -- Results -- " << endl;
      cout << " Energy : " << m_energy << endl;
      cout << " Variance : " << m_variance << endl;
      cout << " Time : " << m_globalTime << endl;
      if (m_system->getGibbsSampling() == false){
        cout << " # Accepted Steps : " << m_globalacceptedSteps << endl;
      }
      cout << endl;
  }
}

void Sampler::computeAverages(double total_time, int numberOfProcesses, int myRank) {
    /* Compute the averages of the sampled quantities.
     */
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();
    int MCcycles=m_system->getNumberOfMetropolisSteps()/numberOfProcesses;
    m_localTime = total_time;
    double norm = 1.0/((double) (MCcycles*numberOfProcesses));     // divided by  number of cycles

    MPI_Reduce(&m_localcumulativeEnergy, &m_globalcumulativeEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&m_localcumulativeEnergy2, &m_globalcumulativeEnergy2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myRank==0){
        m_energy = m_globalcumulativeEnergy *norm;
        m_globalcumulativeEnergy2 = m_globalcumulativeEnergy2 *norm;
        m_globalcumulativeEnergy = m_globalcumulativeEnergy *norm;
        m_variance = (m_globalcumulativeEnergy2 - m_globalcumulativeEnergy*m_globalcumulativeEnergy)*norm;
        m_STD = sqrt(m_variance);
    }

    MPI_Allreduce(m_localaDelta.memptr(), m_globalaDelta.memptr(), nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localbDelta.memptr(), m_globalbDelta.memptr(), nh, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localwDelta.memptr(), m_globalwDelta.memptr(), nx*nh, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(m_localEaDelta.memptr(), m_globalEaDelta.memptr(), nx, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localEbDelta.memptr(), m_globalEbDelta.memptr(), nh, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(m_localEwDelta.memptr(), m_globalEwDelta.memptr(), nx*nh, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //Fra her //
    m_globalcumulativeEnergy /= MCcycles;

    MPI_Reduce(&m_localacceptedSteps, &m_globalacceptedSteps, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&m_localTime, &m_globalTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


    m_globalaDelta /= MCcycles;
    m_globalbDelta /= MCcycles;
    m_globalwDelta /= MCcycles;

    m_globalEaDelta /= MCcycles;
    m_globalEbDelta /= MCcycles;
    m_globalEwDelta /= MCcycles;

    // Compute gradients
    m_agrad = 2*(m_globalEaDelta - m_globalcumulativeEnergy*m_globalaDelta);
    m_bgrad = 2*(m_globalEbDelta - m_globalcumulativeEnergy*m_globalbDelta);
    m_wgrad = 2*(m_globalEwDelta - m_globalcumulativeEnergy*m_globalwDelta);

    // Optimizer parameters (choose either stochastic gradient descent (SGD) or adaptive SGD (ASGD))
    if (m_system->getOptimizer()){
        m_system->getNetwork()->StochasticGradientDescent(m_agrad, m_bgrad, m_wgrad);
    }

    else{
        m_system->getNetwork()->GradientDescent(m_agrad, m_bgrad, m_wgrad);
    }

    m_globalacceptedSteps *= norm;

}


void Sampler::Blocking(int MCcycle, int numberOfProcesses, int myRank){

  double norm = 1.0/((double) (MCcycle));  // divided by  number of local cycles
  double Energy = m_localcumulativeEnergy * norm;
  m_Blocking((MCcycle-1)*numberOfProcesses + myRank) = Energy;
}

void Sampler::Energies(int OptCycles, int myRank){
  if (myRank == 0){
    m_Energies(OptCycles) = m_energy;

    m_Times(OptCycles) = m_globalTime;
  }

}


void Sampler::WriteBlockingtoFile(ofstream& ofile, int myRank){
  vec globalEnergies;
  int MCcycles=m_system->getNumberOfMetropolisSteps();
  globalEnergies.zeros(MCcycles);
  MPI_Allreduce(m_Blocking.memptr(), globalEnergies.memptr(), MCcycles, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (myRank==0){
    for (int i = 0; i < MCcycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Mean energy
      ofile << setw(15) << setprecision(8) << globalEnergies(i) << endl; // Mean energy
    }
  }
 ofile.close();
}
