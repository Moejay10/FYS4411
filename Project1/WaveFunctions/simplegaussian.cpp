#include "simplegaussian.h"
#include <cmath>
#include <cassert>
#include "wavefunction.h"
#include "../system.h"
#include "../particle.h"
#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;

SimpleGaussian::SimpleGaussian(System* system, double alpha, double beta, double gamma, double a) :
        WaveFunction(system) {
    assert(alpha >= 0);
    assert(beta >= 0);
    assert(gamma >= 0);
    assert(a >= 0);
    m_numberOfParameters = 4;
    m_parameters.reserve(4);
    m_parameters.push_back(alpha);
    m_parameters.push_back(beta);
    m_parameters.push_back(gamma);
    m_parameters.push_back(a);

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
     double r2;
     double alpha = m_parameters[0];
     double beta  = m_parameters[1];
     double a = m_parameters[3];
     double wavefunction;

    r2 = 0;
    for (int i = 0; i < N; i++){
      for (int j = 0; j < Dim; j++){
        if (j == 2){
            r2 += beta*m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
          }
        else{
            r2 += m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
          }
        }
      }
    wavefunction = exp(-alpha*r2);

    double diff_ik = 0;
    double correlation = 1;

    vec r_ik(Dim);

    if (m_system->getRepulsivePotential()){
      for (int i = 0; i < N; i++){
        for (int k = N-1; k > i; k--){
          for (int d = 0; d < Dim; d++){
            r_ik(d) = m_system->getParticles()[i]->getPosition()[d] - m_system->getParticles()[k]->getPosition()[d];
            diff_ik += r_ik(d)*r_ik(d);
          }
          diff_ik = sqrt(diff_ik);
          if(diff_ik > a){
              correlation *= 1 - a/diff_ik;
          }
          else{
            correlation = 0;
          }
        }
      }
    }

    return wavefunction*correlation;
}

double SimpleGaussian::derivativeWavefunction(std::vector<class Particle*> particles) {
    /* You need to implement a Gaussian wave function here. The positions of
     * the particles are accessible through the particle[i].getPosition()
     * function.
     *
     * For the actual expression, use exp(-alpha * r^2), with alpha being the
     * (only) variational parameter.
     */
     int Dim = m_system->getNumberOfDimensions();
     int N = m_system->getNumberOfParticles(); // Number of Particles
     double r2;
     double alpha = m_parameters[0];
     double beta  = m_parameters[1];
     double derivative_wavefunction;

    r2 = 0;
    for (int i = 0; i < N; i++){
      for (int j = 0; j < Dim; j++){
        if (j == 2){
            r2 += beta*m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
          }
        else{
            r2 += m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
          }
        }
      }
    derivative_wavefunction = -r2;

    return derivative_wavefunction;
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
     double r2;
     double alpha = m_parameters[0];
     double beta = m_parameters[1];
     double nabla2;

     r2 = 0;
     for (int i = 0; i < N; i++){
       for (int j = 0; j < Dim; j++){
         if (j == 2){
             r2 += beta*beta*m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
           }
         else{
             r2 += m_system->getParticles()[i]->getPosition()[j]*m_system->getParticles()[i]->getPosition()[j];
           }
         }
       }

       nabla2 = -2*alpha*alpha*r2;

       nabla2 += N*((Dim-1)+beta)*alpha;


      return nabla2;
}


double SimpleGaussian::computeFirstDerivativeCorrelation(double dist){
  double a = m_parameters[3];
  return a/(dist*dist - dist*a);
}

double SimpleGaussian::computeDoubleDerivativeCorrelation(double dist){
  double a = m_parameters[3];
  double denominator = a*(2*dist - a);
  double numerator = dist*dist*(dist - a)*(dist - a);
  double output = denominator/numerator;
  return output;
}


double SimpleGaussian::computeDoubleDerivativeInteraction(std::vector<class Particle*> particles) {
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
     double beta = m_parameters[1];
     double a = m_parameters[3];

     double r, temp;
     double r_k, r_i, r_j;
     double temp_k, temp_i, temp_j;

     vec r_ij(Dim);
     vec r_ik(Dim);
     vec force(Dim);

     double diff_ij, diff_ik = 0;
     double term1, term2, term3;
     double total_sum, sum, sum1, sum2, sum3 = 0;


     for (int i = 0; i < N-1; i++){

       r = 0;
       for (int d = 0; d < Dim; d++){
         if (d == 2){
           force(d) = -4*alpha*beta*m_system->getParticles()[i]->getPosition()[d];
         }
         else{
           force(d) = -4*alpha*m_system->getParticles()[i]->getPosition()[d];
         }
       }


     term1 = 0;
     term2 = 0;
     for (int j = N-1; j > i; j--){
       for (int d = 0; d < Dim; d++){
            r_ij(d) = m_system->getParticles()[i]->getPosition()[d] - m_system->getParticles()[j]->getPosition()[d];
            diff_ij += r_ij(d)*r_ij(d);
            term1 += force(d)*r_ij(d);
         }
         diff_ij = sqrt(diff_ij);
         sum1 = term1/diff_ij * computeFirstDerivativeCorrelation(diff_ij);
       }

       diff_ij = 0;
       diff_ik = 0;
       for (int j = N-1; j > i; j--){
         for (int k = N-1; k > i; k--){
           for (int d = 0; d < Dim; d++){
             r_ij(d) = m_system->getParticles()[i]->getPosition()[d] - m_system->getParticles()[j]->getPosition()[d];
             r_ik(d) = m_system->getParticles()[i]->getPosition()[d] - m_system->getParticles()[k]->getPosition()[d];
             diff_ij += r_ij(d)*r_ij(d);
             diff_ik += r_ik(d)*r_ik(d);
             term2   += r_ij(d)*r_ik(d);
           }
           diff_ik = sqrt(diff_ik);
           diff_ij = sqrt(diff_ij);
           sum2 = term2/(diff_ij*diff_ik) * computeFirstDerivativeCorrelation(diff_ij)*computeFirstDerivativeCorrelation(diff_ik);
         }
       }

       diff_ij = 0;
       for (int j = N-1; j > i; j--){
         for (int d = 0; d < Dim; d++){
           r_ij(d) = m_system->getParticles()[i]->getPosition()[d]- m_system->getParticles()[j]->getPosition()[d];
           diff_ij += r_ij(d)*r_ij(d);
         }
         diff_ij = sqrt(diff_ij);
         sum3 = computeDoubleDerivativeCorrelation(diff_ij) + 2/diff_ij * computeFirstDerivativeCorrelation(diff_ij);

       }

       total_sum += sum1 + sum2 + sum3;
       //cout << sum1 << " " << sum2 << " " << sum3 << " " << total_sum << endl;
     }
    //total_sum = sum1 + sum2 + sum3;
    total_sum = -0.5*total_sum;

    return total_sum;
}


/*
        for(int i = 0; i < N; i++){
            sum = 0;

            for(int d = 0; d < Dim; d++){
                if(d==2){
                    sum += beta*beta*m_system->getParticles()[i]->getPosition()[d]*m_system->getParticles()[i]->getPosition()[d] - beta/(2*alpha);
                }
                else{
                    sum += m_system->getParticles()[i]->getPosition()[d]*m_system->getParticles()[i]->getPosition()[d] - 1/(2*alpha);
                }
            }

            sum = 4*alpha*alpha*sum;

            term1 = 0;
            for(int j = N-1; j > i; j--) {
                for(int d = 0; d < Dim; d++){
                    r_ij(d) = m_system->getParticles()[i]->getPosition()[d] - m_system->getParticles()[j]->getPosition()[d];
                    diff_ij += r_ij(d)*r_ij(d);
                    if(d==2){
                        term1 += m_system->getParticles()[i]->getPosition()[d] * r_ij(d) * beta;
                    }
                    else{
                        term1 += m_system->getParticles()[i]->getPosition()[d] * r_ij(d);
                    }


                }
                diff_ij = sqrt(diff_ij);

                double u_der_ij = computeFirstDerivativeCorrelation(diff_ij);
                cout << u_der_ij << endl;
                sum -= 4*alpha*term1*u_der_ij/diff_ij;

                term2 = 0;
                for(int k = N-1; k > i; k--) {
                    for(int d = 0; d < Dim; d++){
                        r_ik(d) = m_system->getParticles()[i]->getPosition()[d] - m_system->getParticles()[k]->getPosition()[d];
                        diff_ik += r_ik(d)*r_ik(d);
                        term2 += r_ik(d)*r_ij(d);
                    }
                    diff_ik = sqrt(diff_ik);

                    sum += term2/(diff_ij*diff_ik) *u_der_ij * computeFirstDerivativeCorrelation(diff_ik);
                }

                sum += computeDoubleDerivativeCorrelation(diff_ij) + 2*u_der_ij/diff_ik; // u''(r_ij) is nan
                //cout << computeFirstDerivativeCorrelation(diff_ij) << endl;

            }

            total_sum += -0.5*sum;
        }


    return total_sum;
}
*/


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

     double h = m_system->getStepSize();
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
     kineticenergy = (1.0/psi)*(-0.5*nabla2/hh);

     return kineticenergy;
}