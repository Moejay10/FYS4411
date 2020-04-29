#pragma once
#include <vector>

class Network {
public:
    Network(class System* system);
    void setWeights(const double &weights);
    void adjustWeights(double change, int dimension);
    void setBiasA(const std::vector<double> &biasA);
    void adjustBiasA(double change, int dimension);
    void setBiasB(const std::vector<double> &biasB);
    void adjustBiasB(double change, int dimension);
    double getWeigths() { return m_weights; }
    std::vector<double> getPositions() { return m_weights; }
    std::vector<double> getBiasA() { return m_biasA; }
    std::vector<double> getBiasB() { return m_biasB; }

protected:
    class System* m_system = nullptr;
};
