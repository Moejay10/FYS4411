#include "neuralnetwork.h"
#include <cassert>
#include <iostream>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"

NeuralNetwork::NeuralNetwork(System* system, double eta) :
          Network(system){
          m_eta = eta;
}

Eigen::VectorXd NeuralNetwork::computeBiasAgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double temp1 = 0;
    double Q;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    Eigen::VectorXd agradient;
    agradient.resize(nx);

    for (int m = 0; m < nx; m++){
      temp1 = (getPositions()[m] - getBiasA()[m])/(sigma2);
      agradient(m) = temp1;
    }

    return agradient;
}

Eigen::VectorXd NeuralNetwork::computeBiasBgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double temp1 = 0;
    double Q;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    Eigen::Eigen::VectorXd
    bgradient.resize(nh);

    for (int n = 0; n < nh; n++){
      temp1 = 0;
      for (int i = 0; i < nx; i++){
        temp1 += (getPositions()[i] * getWeigths()[i][n]);
      }
      Q = exp(-getBiasB()[n] - temp1/(sigma2)) + 1;
      bgradient(n) = 1/Q;
    }

    return bgradient;

}


Eigen::VectorXd NeuralNetwork::computeWeightsgradients() {
    // Here we compute the derivative of the wave function
    double sigma = m_system->getWaveFunction()->getParameters()[0];
    double sigma2 = sigma*sigma;
    double temp1 = 0;
    double Q;
    int nx = m_system->getNumberOfInputs();
    int nh = m_system->getNumberOfHidden();

    Eigen::VectorXd wgradient;
    wgradient.resize(nx*nh);

    for (int m = 0; m < nx; m++){
      for (int n = 0; n < nh; n++){
        temp1 = 0;
        for (int i = 0; i < nx; i++){
          temp1 += (getPositions()[i] * getWeigths()[i][n]);
        }
        Q = exp(-getBiasB()[n] - temp1/(sigma2)) + 1;
        wgradient(m*nx + n) = getPositions()[m]/(sigma2*Q);
      }
    }

    return wgradient;
}

void NeuralNetwork::optimizeWeights(Eigen::VectorXd agrad, Eigen::VectorXd bgrad, Eigen::VectorXd wgrad){
  // Compute new parameters
  int nx = m_system->getNumberOfInputs();
  int nh = m_system->getNumberOfHidden();

  for (int i = 0; i < nx; i++){
      m_biasA[i] = m_biasA[i] - m_eta*agrad(i);
  }

  for (int j = 0; j < nh; j++){
      m_biasB[j] = m_biasB[j] - m_eta*bgrad(j);
  }

  for (int i = 0; i < nx; i++){
      for (int j = 0; j < nh; j++){
          m_weights[i*nx + j] = m_weights[i*nx + j] - m_eta*wgrad(i*nx + j);
      }
  }

}


void NeuralNetwork::setPositons(const std::vector<double> &positions) {
    m_positions = positions;
}

void NeuralNetwork::adjustPositions(double change, int dimension, int input) {
    int n = m_system->getNumberOfParticles();
    m_positions[input*n + dimension] += change;
}

void NeuralNetwork::setWeights(std::vector<double> &weights) {
    m_weights = weights;
}


void NeuralNetwork::setBiasA(std::vector<double> &biasA) {
    m_biasA = biasA;
}

void NeuralNetwork::setBiasB(std::vector<double> &biasB) {
    m_biasB = biasB;
}
