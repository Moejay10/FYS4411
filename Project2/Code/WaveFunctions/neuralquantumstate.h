#pragma once
#include "wavefunction.h"
#include <armadillo>
#include <vector>


using namespace arma;

class NeuralQuantumState : public WaveFunction {
public:
    NeuralQuantumState(class System* system, double sigma, double gibbs);
    double evaluate(vec Q);
    vec computeFirstDerivative(vec Q);
    vec computeDoubleDerivative(vec Q);
    vec computeQ();
};
