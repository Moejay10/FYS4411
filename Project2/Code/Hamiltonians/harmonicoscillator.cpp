#include "harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include <armadillo>
#include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;
using namespace arma;

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




double HarmonicOscillator::computeLocalEnergy(NeuralNetwork* neuralnetwork) {
    // Here we compute the analytical local energy
    int M = m_system->getNumberOfInputs();
    double kineticenergy = potentialenergy = 0, totalenergy, firstder, secondder, x;

    for (int m = 0; m < M; m++){
      firstder = m_system->getWaveFunction()->computeFirstDerivative(neuralnetwork, m);
      secondder = m_system->getWaveFunction()->computeDoubleDerivative(neuralnetwork, m);
      x = m_system->getNeuralNetwork()->getPositions()[m];
      kineticenergy += (-(temp*temp) - secondder;
      potentialenergy += m_omega*m_omega*x*x;
    }

    totalenergy = 0.5*(kineticenergy + potentialenergy);
}
