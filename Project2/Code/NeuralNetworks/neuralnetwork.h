#pragma once
#include "network.h"
#include <vector>
#include <armadillo>
using namespace arma;

class NeuralNetwork : public Network {
public:
    NeuralNetwork(class System* system, double eta, double a, double A, double asgdOmega, double fmax, double fmin,
               double t0, double t1, int numberOfInputs, int numberOfHidden);

    vec computeBiasAgradients();
    vec computeBiasBgradients(vec Q);
    vec computeWeightsgradients(vec Q);

    void GradientDescent(vec agrad, vec bgrad, vec wgrad);
    void StochasticGradientDescent(vec agrad, vec bgrad, vec wgrad);


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
    double     m_a = 0;
    double     m_A = 0;
    double     m_asgdOmega = 0;
    double     m_fmax = 0;
    double     m_fmin = 0;
    double     m_t = 0;
    double     m_tprev = 0;

    vec m_gradPrevBiasA;
    vec m_gradPrevBiasB;
    vec m_gradPrevWeights;

    mat m_weights;
    vec m_positions;
    vec m_biasA;
    vec m_biasB;
};
