#include <vector>

class Network {
public:
    Network(class System* system);
    void setWeights(const std::vector<double> &weights);
    void adjustWeights(double change, int dimension);
    void setBiasA(const std::vector<double> &biasA);
    void adjustBiasA(double change, int dimension);
    void setBiasB(const std::vector<double> &biasB);
    void adjustBiasB(double change, int dimension);
    void setPositions(const std::vector<double> &positions);
    void adjustPositions(double change, int dimension, int input);

    std::vector<double> getWeigths() { return m_weights; }
    std::vector<double> getPositions() { return m_weights; }
    std::vector<double> getBiasA() { return m_biasA; }
    std::vector<double> getBiasB() { return m_biasB; }

private:
  std::vector<double> m_positions = std::vector<double>();
  std::vector<double> m_biasA = std::vector<double>();
  std::vector<double> m_biasB = std::vector<double>();
  std::vector<double> m_weights = std::vector<double>();

protected:
    class System* m_system = nullptr;
};
