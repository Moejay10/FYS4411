#include <cassert>
#include <iostream>
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"
#include "../NeuralNetworks/network.h"

#include "harmonicoscillator.h"

using std::cout;
using std::endl;

HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
        Hamiltonian(system) {
    assert(omega > 0);
    m_omega  = omega;
}

/* In this class, we are computing the energies. Note that
 * when using numerical differentiation, the computation of the kinetic
 * energy becomes the same for all Hamiltonians, and thus the code for
 * doing this should be moved up to the super-class, Hamiltonian.
 * (Great tip that should be implemented in the future)
 */




double HarmonicOscillator::computeLocalEnergy(Network* network) {
    // Here we compute the analytical local energy
    int M = m_system->getNumberOfInputs();
    double kineticenergy = 0;
    double potentialenergy = 0;
    double totalenergy, firstder, secondder, x;

    for (int m = 0; m < M; m++){
      firstder = m_system->getWaveFunction()->computeFirstDerivative(network, m);
      secondder = m_system->getWaveFunction()->computeDoubleDerivative(network, m);
      x = network->getPositions()[m];
      kineticenergy += (-(firstder*firstder) - secondder);
      potentialenergy += m_omega*m_omega*x*x;
    }

    totalenergy = 0.5*(kineticenergy + potentialenergy);

    return totalenergy;
}
