#pragma once
#include <vector>
#include "../NeuralNetworks/network.h"


class WaveFunction {
public:
    WaveFunction(class System* system);
    int     getNumberOfParameters() { return m_numberOfParameters; }
    std::vector<double> getParameters() { return m_parameters; }
    virtual double evaluate(Network* network) = 0;
    virtual double computeDoubleDerivative(Network* network, int m) = 0;
    virtual double computeFirstDerivative(Network* network, int m) = 0;




protected:
    int     m_numberOfParameters = 0;
    std::vector<double> m_parameters = std::vector<double>();
    class System* m_system = nullptr;
};
