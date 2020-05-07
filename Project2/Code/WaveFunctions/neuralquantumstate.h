#pragma once
#include "wavefunction.h"

class NeuralQuantumState : public WaveFunction {
public:
    NeuralQuantumState(class System* system, double sigma);
    double NeuralQuantumState::evaluate(NeuralNetwork* neuralnetwork);
    double computeFirstDerivative(NeuralNetwork* neuralnetwork, int m);
    double computeDoubleDerivative(NeuralNetwork* neuralnetwork, int m);
};
