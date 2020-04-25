#pragma once
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork();
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


private:
    int     m_numberOfDimensions = 0;
    double m_weights;
    std::vector<double> m_positions = std::vector<double>();
    std::vector<double> m_biasA = std::vector<double>();
    std::vector<double> m_biasB = std::vector<double>();

};
