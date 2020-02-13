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
     double temp, r;
     double alpha = m_parameters[0];
     double psi_T = 0;

    for (int i = 0; i < N; i++){
       r = 0;
       for (int j = 0; j < Dim; j++){
          temp = m_system->getParticles()[i]->getPosition()[j];
          r += temp*temp; // x^2 + y^2 + z^2
       }
      r = sqrt(r); // sqrt(x^2 + y^2 + z^2)
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
     int Dim = m_system->getNumberOfDimensions(); // The Dimension
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double r, temp;
     double alpha = m_parameters[0];
     double psi_T = evaluate(particles);
     double factor = 0;

     for (int i = 0; i < N; i++){
       for (int j = 0; j < Dim; j++){
         temp = m_system->getParticles()[i]->getPosition()[j];
         r += temp*temp; // x^2 + y^2 + z^2
       }
      r = sqrt(r); // sqrt(x^2 + y^2 + z^2)
      factor += (-2*alpha + 4*alpha*alpha*r*r);
     }

     double nabla2 = factor;
    return nabla2;
}

double SimpleGaussian::computeDoubleDerivativeNumerical(std::vector<class Particle*> particles) {
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
     double r_plus, r_minus, temp_plus, temp_minus;
     double alpha = m_parameters[0];
     double psi_plus, psi_minus;
     double psi = evaluate(particles); // psi(r)
     double factor = 0;

     double h = 0.01;
     double hh= h*h;

     for (int i = 0; i < N; i++){
       r_plus = 0;
       r_minus = 0;
       for (int j = 0; j < Dim; j++){
         m_particles[i]->adjustPosition(h,j);
         temp_plus = m_system->getParticles()[i]->getPosition()[j];
         r_plus += temp_plus*temp_plus; // x^2 + y^2 + z^2

         m_particles[i]->adjustPosition(-2*h,j);
         temp_minus = m_system->getParticles()[i]->getPosition()[j];
         r_minus += temp_minus*temp_minus; // x^2 + y^2 + z^2

         m_particles[i]->adjustPosition(+h,j);

       }
       r_plus = sqrt(r_plus); // sqrt(x^2 + y^2 + z^2)
       psi_plus *= exp(-alpha*r_plus*r_plus);

       r_minus = sqrt(r_minus); // sqrt(x^2 + y^2 + z^2)
       psi_minus *= exp(-alpha*r_minus*r_minus);

     }

     double nabla2 = (psi_plus - 2*psi + psi_minus)/(hh);


     return nabla2;
}
