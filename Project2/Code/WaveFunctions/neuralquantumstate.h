#pragma once
#include "wavefunction.h"
#include <armadillo>

class NeuralQuantumState : public WaveFunction {
public:
    SimpleGaussian(class System* system, double sigma);
    double NeuralQuantumState::evaluate(NeuralNetwork* neuralnetwork);
    double computeFirstDerivative(NeuralNetwork* neuralnetwork, int m);
    double computeDoubleDerivative(NeuralNetwork* neuralnetwork, int m);
};
