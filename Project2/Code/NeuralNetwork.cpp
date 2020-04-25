#include "particle.h"
#include <cassert>

NeuralNetwork::NeuralNetwork() {
}

void NeuralNetwork::setPositons(const std::vector<double> &positions) {
    m_positions = positions;
}

void NeuralNetwork::adjustWeights(double change, int dimension) {
    m_weights.at(dimension) += change;
}

void NeuralNetwork::setWeights(const double &weights) {
    m_weights = weights;
}

void NeuralNetwork::adjustWeights(double change, int dimension) {
    m_weights.at(dimension) += change;
}

void NeuralNetwork::setBiasA(const std::vector<double> &biasA) {
    m_biasA = biasA;
}

void NeuralNetwork::adjustBiasA(double change, int dimension) {
    m_biasA.at(dimension) += change;
}


void NeuralNetwork::setBiasB(const std::vector<double> &biasB) {
    m_biasB = biasB;
}

void NeuralNetwork::adjustBiasB(double change, int dimension) {
    m_biasB.at(dimension) += change;
}
