#pragma once
#include "wavefunction.h"

class SimpleGaussian : public WaveFunction {
public:
    SimpleGaussian(class System* system, double alpha, double beta, double gamma);
    double evaluate(std::vector<class Particle*> particles);
    double derivativeWavefunction(std::vector<class Particle*> particles);
    double computeDoubleDerivative(std::vector<class Particle*> particles);
    double computeDoubleNumericalDerivative(std::vector<class Particle*> particles);
};
