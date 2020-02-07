#include "harmonicoscillator.h"
#include <cassert>
#include <iostream>
#include "../system.h"
#include "../particle.h"
#include "../WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;

HarmonicOscillator::HarmonicOscillator(System* system, double omega) :
        Hamiltonian(system) {
    assert(omega > 0);
    m_omega  = omega;
}

double HarmonicOscillator::computeLocalEnergy(std::vector<Particle*> particles) {
    /* Here, you need to compute the kinetic and potential energies. Note that
     * when using numerical differentiation, the computation of the kinetic
     * energy becomes the same for all Hamiltonians, and thus the code for
     * doing this should be moved up to the super-class, Hamiltonian.
     *
     * You may access the wave function currently used through the
     * getWaveFunction method in the m_system object in the super-class, i.e.
     * m_system->getWaveFunction()...
     */
    double alpha = m_system->getWaveFunction()->getParameters()[0];
    int N = m_system->getNumberOfParticles(); // Number of Particles
    double r;
    double E_L;
    double derivative = m_system->getWaveFunction()->computeDoubleDerivative(particles);

    for (int i = 0; i < N; i++){
      r = m_system->getParticles()[i]->getPosition()[i];
      E_L += r*r;
    }
    E_L = N*alpha + (0.5 - 2*alpha*alpha)*E_L;
    return E_L;
}
