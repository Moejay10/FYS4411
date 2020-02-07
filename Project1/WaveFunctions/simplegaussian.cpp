#include "simplegaussian.h"
#include <cmath>
#include <cassert>
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"
#include <iostream>
using namespace std;

SimpleGaussian::SimpleGaussian(System* system, double alpha) :
        WaveFunction(system) {
    assert(alpha >= 0);
    m_numberOfParameters = 1;
    m_parameters.reserve(1);
    m_parameters.push_back(alpha);
}

double SimpleGaussian::evaluate(std::vector<class Particle*> particles) {
    /* You need to implement a Gaussian wave function here. The positions of
     * the particles are accessible through the particle[i].getPosition()
     * function.
     *
     * For the actual expression, use exp(-alpha * r^2), with alpha being the
     * (only) variational parameter.
     */
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double r;
     double alpha = m_parameters[0];
     double psi_T = 0;

     for (int i = 0; i < N; i++){
       r = m_system->getParticles()[i]->getPosition()[i];
       psi_T *= exp(-alpha*r*r);
     }

    return psi_T;
}

double SimpleGaussian::computeDoubleDerivative(std::vector<class Particle*> particles) {
    /* All wave functions need to implement this function, so you need to
     * find the double derivative analytically. Note that by double derivative,
     * we actually mean the sum of the Laplacians with respect to the
     * coordinates of each particle.
     *
     * This quantity is needed to compute the (local) energy (consider the
     * Schrödinger equation to see how the two are related).
     */
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double r;
     double alpha = m_parameters[0];
     double psi_T = evaluate(particles);
     double factor = 0;

     for (int i = 0; i < N; i++){
       r = m_system->getParticles()[i]->getPosition()[i];
       factor += (-2*alpha + 4*alpha*alpha*r*r);
     }

     double nabla2 = factor;
    return nabla2;
}
