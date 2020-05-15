#pragma once
#include <armadillo>
#include <vector>
#include "../NeuralNetworks/network.h"
using namespace arma;

class WaveFunction {
public:
    WaveFunction(class System* system);
    int     getNumberOfParameters() { return m_numberOfParameters; }
    std::vector<double> getParameters() { return m_parameters; }
    virtual double evaluate(vec Q) = 0;
    virtual vec computeDoubleDerivative(vec Q) = 0;
    virtual vec computeFirstDerivative(vec Q) = 0;
    virtual vec computeQ() = 0;



protected:
    int     m_numberOfParameters = 0;
    std::vector<double> m_parameters = std::vector<double>();
    class System* m_system = nullptr;
};
