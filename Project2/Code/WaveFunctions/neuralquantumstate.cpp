#include <cmath>
#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../NeuralNetworks/network.h"

#include "neuralquantumstate.h"

using namespace arma;
using namespace std;

NeuralQuantumState::NeuralQuantumState(System* system, double sigma) :
        WaveFunction(system) {
    assert(sigma >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(sigma);
}

double NeuralQuantumState::evaluate() {
    // Here we compute the wave function.
    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double psi1 = 0, psi2 = 1;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec Q = m_system->getNetwork()->getBiasB() + (1.0/sigma2)*(m_system->getNetwork()->getPositions().t()*m_system->getNetwork()->getWeigths()).t();

    for (int i = 0; i < nx; i++){
       psi1 += (m_system->getNetwork()->getPositions()(i) - m_system->getNetwork()->getBiasA()(i)) * (m_system->getNetwork()->getPositions()(i) - m_system->getNetwork()->getBiasA()(i));

    }
     psi1 = exp(-psi1/(2*sigma2));

    for (int j = 0; j < nh; j++) {
        psi2 *= (1 + exp(Q(j)));
    }

    return psi1*psi2;
}

vec NeuralQuantumState::computeFirstDerivative() {
    // Here we compute the first derivative
    //of the wave function with respect to the visible nodes

    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double temp;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec psi1(nx);
    vec Q = m_system->getNetwork()->getBiasB() + (1.0/sigma2)*(m_system->getNetwork()->getPositions().t()*m_system->getNetwork()->getWeigths()).t();

    for (int i = 0; i < nx; i++){
      temp = 0;
      for (int j = 0; j < nh; j++){
        temp += m_system->getNetwork()->getWeigths()(i,j)/(1.0+exp(-Q(j)));
      }
      psi1(i) = -(m_system->getNetwork()->getPositions()(i) - m_system->getNetwork()->getBiasA()(i))/sigma2 + temp/sigma2;

    }

    return psi1;
}

vec NeuralQuantumState::computeDoubleDerivative() {
    // Here we compute the double derivative of the wavefunction
    double sigma = m_parameters[0];
    double sigma2 = sigma*sigma;
    double temp;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec psi2(nx);
    vec Q = m_system->getNetwork()->getBiasB() + (1.0/sigma2)*(m_system->getNetwork()->getPositions().t()*m_system->getNetwork()->getWeigths()).t();

    for (int i = 0; i < nx; i++){
      temp = 0;
      for (int j = 0; j < nh; j++){
        temp += m_system->getNetwork()->getWeigths()(i,j)*m_system->getNetwork()->getWeigths()(i,j)*exp(-Q(j))/(1.0+exp(-Q(j)))*(1.0+exp(-Q(j)));
      }
      psi2(i) = -1.0/sigma2 + temp/(sigma2*sigma2);

    }

    return psi2;
}
