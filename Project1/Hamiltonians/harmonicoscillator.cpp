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
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    double r2, temp;
    double psi = m_system->getWaveFunction()->evaluate(particles); // psi(r)
    double numerical_kineticenergy, potentialenergy;
    double analytical_E_L, numerical_E_L;
    double analytical_kineticenergy;

    potentialenergy = 0;
    for (int i = 0; i < N; i++){
      r2 = 0;
      for (int j = 0; j < Dim; j++){
        temp = m_system->getParticles()[i]->getPosition()[j];
        r2 += temp*temp; // x^2 + y^2 + z^2
      }
      potentialenergy += 0.5*r2;
    }

    numerical_kineticenergy = m_system->getWaveFunction()->computeDoubleNumericalDerivative(particles);
    numerical_E_L = numerical_kineticenergy + potentialenergy;

    analytical_kineticenergy = m_system->getWaveFunction()->computeDoubleDerivative(particles);
    analytical_E_L = analytical_kineticenergy + potentialenergy;
    //analytical_E_L = Dim*N*alpha + (0.5 - 2*alpha*alpha)*sum_r;


    cout << "Numerical Energy = " << numerical_E_L << endl;
    cout << "Analytical Energy = " << analytical_E_L << endl;
    cout << "Numerical Kinetic Energy = " << numerical_kineticenergy<< endl;
    cout << "Analytical Kinetic Energy = " << analytical_kineticenergy<< endl;
    cout << "Potential Energy = " << potentialenergy << endl;


    return numerical_E_L;
}
