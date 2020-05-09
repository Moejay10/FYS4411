#include "neuralnetwork.h"
#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"

using namespace arma;

NeuralNetwork::NeuralNetwork(System* system, double eta) :
          Network(system){
          m_eta = eta;
}

vec NeuralNetwork::computeBiasAgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double temp1 = 0;
    int nx = m_system->getNumberOfInputs();

    vec agradient(nx);

    for (int m = 0; m < nx; m++){
      temp1 = (getPositions()(m) - getBiasA()(m))/(sigma2);
      agradient(m) = temp1;
    }

    return agradient;
}

vec NeuralNetwork::computeBiasBgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    int nh = m_system->getNumberOfHidden();

    vec Q = getBiasB() + (1.0/sigma2)*(getPositions().t()*getWeigths()).t();
    vec bgradient(nh);

    for (int j = 0; j < nh; j++) {
        bgradient(j) = 1.0/(1.0+exp(-Q(j)));
    }

    return bgradient;
}


vec NeuralNetwork::computeWeightsgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    vec Q = getBiasB() + (1.0/sigma2)*(getPositions().t()*getWeigths()).t();
    vec wgradient(nx*nh);

    for (int i = 0; i < nx; i++) {
      for (int j = 0; j < nh; j++) {
          wgradient(i*nh + j) = getPositions()(i)/(sigma2*(1.0+exp(-Q(j))));
      }
    }

    return wgradient;
}

void NeuralNetwork::optimizeWeights(vec agrad, vec bgrad, vec wgrad){
  // Compute new parameters
  int nx = m_system->getNumberOfInputs();
  int nh = m_system->getNumberOfHidden();

  for (int i = 0; i < nx; i++){
      m_biasA(i) = m_biasA(i) - m_eta*agrad(i);
  }

  for (int j = 0; j < nh; j++){
      m_biasB(j) = m_biasB(j) - m_eta*bgrad(j);
  }

  for (int i = 0; i < nx; i++){
      for (int j = 0; j < nh; j++){
          m_weights(i,j) = m_weights(i,j) - m_eta*wgrad(i*nh + j);
      }
  }

}


void NeuralNetwork::setPositions(const vec &positions) {
    assert(positions.size() == m_system->getNumberOfInputs());
    m_positions = positions;
}

void NeuralNetwork::adjustPositions(double change, int dimension, int input) {
    int n = m_system->getNumberOfParticles();
    m_positions(input*n + dimension) += change;
}

void NeuralNetwork::setWeights(mat &weights) {
    m_weights = weights;
}


void NeuralNetwork::setBiasA(vec &biasA) {
    m_biasA = biasA;
}

void NeuralNetwork::setBiasB(vec &biasB) {
    m_biasB = biasB;
}
