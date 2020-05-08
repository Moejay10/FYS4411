#pragma once
#include <vector>
#include <armadillo>
using namespace arma;

class Network {
public:
    Network(class System* system);

    virtual vec computeBiasAgradients() = 0;
    virtual vec computeBiasBgradients() = 0;
    virtual vec computeWeightsgradients() = 0;

    virtual void optimizeWeights(vec agrad, vec bgrad, vec wgrad) = 0;

    void setPositions(const std::vector<double> &positions);
    void adjustPositions(double change, int dimension, int input);
    void setWeights(std::vector<double> &weights);
    void setBiasA(std::vector<double> &biasA);
    void setBiasB(std::vector<double> &biasB);

    std::vector<double> getPositions() { return m_positions; }
    std::vector<double> getWeigths() { return m_weights; }
    std::vector<double> getBiasA() { return m_biasA; }
    std::vector<double> getBiasB() { return m_biasB; }

protected:
  std::vector<double> m_weights = std::vector<double>();
  std::vector<double> m_positions = std::vector<double>();
  std::vector<double> m_biasA = std::vector<double>();
  std::vector<double> m_biasB = std::vector<double>();

  class System* m_system = nullptr;
};