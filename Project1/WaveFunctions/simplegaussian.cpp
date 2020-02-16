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
     int Dim = m_system->getNumberOfDimensions();
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double temp, r2, sum_r2;
     double alpha = m_parameters[0];
     double wavefunction;

    for (int i = 0; i < N; i++){
      r2 = 0;
      for (int j = 0; j < Dim; j++){
         temp = m_system->getParticles()[i]->getPosition()[j];
         r2 += temp*temp; // x^2 + y^2 + z^2
         }
      sum_r2 += r2;
      }
    wavefunction = exp(-alpha*sum_r2);

    return wavefunction;
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
     int Dim = m_system->getNumberOfDimensions(); // The Dimension
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double r2, temp;
     double alpha = m_parameters[0];
     double psi_T = evaluate(particles);
     double factor, nabla2;

     nabla2 = 0;
     //factor = -0.5*(-2*Dim*alpha + 4*alpha*alpha);
     for (int i = 0; i < N; i++){
       r2 = 0;
       for (int j = 0; j < Dim; j++){
         temp = m_system->getParticles()[i]->getPosition()[j];
         r2 += temp*temp; // x^2 + y^2 + z^2
         }
       nabla2 += r2;
       }

    nabla2 *= -2*alpha*alpha;
    nabla2 += N*Dim*alpha; // Why not multiply with N?

    return nabla2;
}

double SimpleGaussian::computeDoubleNumericalDerivative(std::vector<class Particle*> particles) {
    /* All wave functions need to implement this function, so you need to
     * find the double derivative analytically. Note that by double derivative,
     * we actually mean the sum of the Laplacians with respect to the
     * coordinates of each particle.
     *
     * This quantity is needed to compute the (local) energy (consider the
     * Schrödinger equation to see how the two are related).
     */
     int Dim = m_system->getNumberOfDimensions(); // The Dimension
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double alpha = m_parameters[0];
     double psi_plus, psi_minus = 0;
     double psi = evaluate(particles); // psi(r)
     double kineticenergy, nabla2;

     double h = 1e-4;
     double hh= h*h;

     nabla2 = 0;
     for (int i = 0; i < N; i++){
       for (int j = 0; j < Dim; j++){
         particles[i]->adjustPosition(h,j);
         psi_plus = evaluate(particles);

         particles[i]->adjustPosition(-2*h,j);
         psi_minus = evaluate(particles);

         particles[i]->adjustPosition(h,j);

         nabla2 += (psi_plus - 2*psi + psi_minus);
       }

     }
     kineticenergy = (1.0/psi)*(-0.5*nabla2/hh); //- Dim*(N-1)*alph; // Why substract to work?

     return kineticenergy;
}
