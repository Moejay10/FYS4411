#pragma once
#include "wavefunction.h"

class NeuralQuantumState : public WaveFunction {
public:
    NeuralQuantumState(class System* system, double sigma);
    double evaluate(Network* network);
    double computeFirstDerivative(Network* network, int m);
    double computeDoubleDerivative(Network* network, int m);
};
