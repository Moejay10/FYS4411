#pragma once
#include <vector>
#include "../NeuralNetworks/network.h"

class Hamiltonian {
public:
    Hamiltonian(class System* system);
    virtual double computeLocalEnergy(Network* network) = 0;
    //virtual std::vector<double> computeQuantumForce(std::vector<class Particle*> particles, int i) = 0;

protected:
    class System* m_system = nullptr;

};
