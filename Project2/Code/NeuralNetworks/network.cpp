#include "network.h"

Network::Network(System* system){
  m_system = system;
}

void Network::setPositions(const std::vector<double> &positions) {
    assert(positions.size() == m_system->getNumberOfInputs());
    m_positions = positions;
}

void Network::adjustPositions(double change, int dimension, int input) {
    int n = m_system->getNumberOfParticles();
    m_positions.at(input*n + dimension) += change;
}

void Network::setWeights(std::vector<double> &weights) {
    m_weights = weights;
}


void Network::setBiasA(std::vector<double> &biasA) {
    m_biasA = biasA;
}

void Network::setBiasB(std::vector<double> &biasB) {
    m_biasB = biasB;
}
