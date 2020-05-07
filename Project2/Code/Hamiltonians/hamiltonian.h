#include <vector>

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computeLocalEnergy(NeuralNetwork* neuralnetwork) = 0;
    //virtual std::vector<double> computeQuantumForce(std::vector<class Particle*> particles, int i) = 0;

protected:
    class System* m_system = nullptr;
};
