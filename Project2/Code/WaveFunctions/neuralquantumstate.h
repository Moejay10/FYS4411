
#include "wavefunction.h"

class NeuralQuantumState : public WaveFunction {
public:
    NeuralQuantumState(class System* system, double sigma);
    double evaluate(Network* neuralnetwork);
    double computeFirstDerivative(Network* neuralnetwork, int m);
    double computeDoubleDerivative(Network* neuralnetwork, int m);
};
