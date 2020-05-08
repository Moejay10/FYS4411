#include "neuralquantumstate.h"
#include <cmath>
#include <cassert>
#include "wavefunction.h"
#include "../system.h"
#include <iostream>
using namespace std;

NeuralQuantumState::NeuralQuantumState(System* system, double sigma) :
        WaveFunction(system) {
    assert(sigma >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(sigma);


}

double NeuralQuantumState::evaluate(Network* network) {
     // Here we compute the wave function.
     double sigma = m_parameters[0];
     double sigma2 = sigma*sigma;
     double psi1 = 0, psi2 = 1, temp1 = 0;
     int nx = m_system->getNumberOfInputs();
     int nh = m_system->getNumberOfHidden();


     for (int i = 0; i < nx; i++){
       psi1 += (network->getPositions()[i] - network->getBiasA()[i]) * (network->getPositions()[i] - network->getBiasA()[i]);
     }
     psi1 = exp(-psi1/(2*sigma2));

     // Can possibly implement single for loops for speedups.
     for (int j = 0; j < nh; j++){
       temp1 = 0;
       for (int i = 0; i < nx; i++){
         temp1 += (network->getPositions()[i] * network->getWeigths()[i*nx +j])/(sigma2);
       }
       psi2 *= 1 + exp(network->getBiasB()[j] + temp1);
     }

     return psi1*psi2;
}

double NeuralQuantumState::computeFirstDerivative(Network* network, int m) {
    // Here we compute the derivative of the wave function
    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double psi2 = 0, temp1 = 0;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    for (int j = 0; j < nh; j++){
      temp1 = 0;
      for (int i = 0; i < nx; i++){
        temp1 += (network->getPositions()[i] * network->getWeigths()[i*nx + j]);
      }
      psi2 += network->getWeigths()[m*nx + j]/(1 + exp(-network->getBiasB()[j] - temp1/(sigma2) ) );
    }

    psi2 -= network->getPositions()[m] - network->getBiasA()[m];

    psi2 /= sigma2;

    return psi2;
}

double NeuralQuantumState::computeDoubleDerivative(Network* network, int m) {
    // Here we compute the double derivative of the wavefunction
    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double psi2 = 1, temp1 = 0;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    for (int j = 0; j < nh; j++){
      temp1 = 0;
      for (int i = 0; i < nx; i++){
        temp1 += (network->getPositions()[i] * network->getWeigths()[i*nx + j]);
      }
      double Q = exp(network->getBiasB()[j] + temp1/(sigma2));
      double w = network->getWeigths()[m*nx + j];
      psi2 += w*w * Q/((Q+1)*(Q+1));
    }

    psi2 /= (sigma2*sigma2);

    psi2 -= 1/(sigma2);

    return psi2;
}
