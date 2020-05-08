#pragma once
#include "network.h"
#include <vector>
#include <armadillo>
using namespace arma;

class NeuralNetwork : public Network {
public:
    NeuralNetwork(class System* system, double eta);

    vec computeBiasAgradients();
    vec computeBiasBgradients();
    vec computeWeightsgradients();

    void optimizeWeights(vec agrad, vec bgrad, vec wgrad);



private:
    double     m_eta = 0;

    std::vector<double> m_weights = std::vector<double>();
    std::vector<double> m_positions = std::vector<double>();
    std::vector<double> m_biasA = std::vector<double>();
    std::vector<double> m_biasB = std::vector<double>();
};
