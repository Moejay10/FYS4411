#pragma once
#include "network.h"
#include <vector>

class NeuralNetwork : public Network {
public:
    NeuralNetwork(class System* system, double eta);

    Eigen::VectorXd computeBiasAgradients();
    Eigen::VectorXd computeBiasBgradients();
    Eigen::VectorXd computeWeightsgradients();

    void optimizeWeights(std::vector<double> agrad, std::vector<double> bgrad, std::vector<double> wgrad);


    void setPositions(const std::vector<double> &positions);
    void adjustPositions(double change, int dimension, int input);
    void setWeights(std::vector<double> &weights);
    void setBiasA(std::vector<double> &biasA);
    void setBiasB(std::vector<double> &biasB);
    std::vector<double> getPositions() { return m_positions; }
    std::vector<double> getWeigths() { return m_weights; }
    std::vector<double> getBiasA() { return m_biasA; }
    std::vector<double> getBiasB() { return m_biasB; }


private:
    int     m_numberOfDimensions = 0;
    double     m_eta = 0;
    std::vector<double> m_weights = std::vector<double>();
    std::vector<double> m_positions = std::vector<double>();
    std::vector<double> m_biasA = std::vector<double>();
    std::vector<double> m_biasB = std::vector<double>();

};
