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

    void setPositions(const vec &positions);
    void adjustPositions(double change, int dimension, int input);
    void setWeights(mat &weights);
    void setBiasA(vec &biasA);
    void setBiasB(vec &biasB);

    vec getPositions() { return m_positions; }
    mat getWeigths() { return m_weights; }
    vec getBiasA() { return m_biasA; }
    vec getBiasB() { return m_biasB; }

private:
    double     m_eta = 0;

    mat m_weights;
    vec m_positions;
    vec m_biasA;
    vec m_biasB;
};
