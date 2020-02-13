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
    double r, temp, sum_r, sum_r2;
    double psi = m_system->getWaveFunction()->evaluate(particles); // psi(r)
    double analytical_E_L;
    double analytical_derivative = m_system->getWaveFunction()->computeDoubleDerivative(particles);


    for (int i = 0; i < N; i++){
      r = 0;
      for (int j = 0; j < Dim; j++){
        temp = m_system->getParticles()[i]->getPosition()[j];
        r += temp*temp; // x^2 + y^2 + z^2
      }
      sum_r += r;
      sum_r2 += r*0.5;
    }
    analytical_E_L = Dim*N*alpha + (0.5 - 2*alpha*alpha)*sum_r;

    double kineticenergy = m_system->getWaveFunction()->computeDoubleNumericalDerivative(particles);
    double potentialenergy = 0.5*sum_r;
    double numerical_E_L = kineticenergy + potentialenergy;

/*
    cout << "Numerical Energy " << numerical_E_L << endl;
    cout << "Analytical Energy " << analytical_E_L << endl;
    cout << "Numerical Kinetic Energy " << kineticenergy + potentialenergy<< endl;
    cout << "Numerical Potential Energy " << potentialenergy << endl;
*/


    return numerical_E_L;
}
