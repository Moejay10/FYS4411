#include <iostream>
#include <cmath>
#include <vector>
#include "sampler.h"
#include "system.h"
#include "particle.h"
#include "Hamiltonians/hamiltonian.h"
#include "WaveFunctions/wavefunction.h"

using std::cout;
using std::endl;


Sampler::Sampler(System* system) {
    m_system = system;
    m_stepNumber = 0;
}

void Sampler::setNumberOfMetropolisSteps(int steps) {
    m_numberOfMetropolisSteps = steps;
}

void Sampler::sample(ofstream& ofile, bool acceptedStep, int MCcycles) {
    // Make sure the sampling variable(s) are initialized at the first step.
    if (m_stepNumber == 0) {
        m_cumulativeEnergy = 0;
    }


    /* Here you should sample all the interesting things you want to measure.
     * Note that there are (way) more than the single one here currently.
     */
    double localEnergy = m_system->getHamiltonian()->
                         computeLocalEnergy(m_system->getParticles());
    //cout << "Local Energy = " << localEnergy << endl;
    m_cumulativeEnergy  += localEnergy;
    m_cumulativeEnergy2  += localEnergy*localEnergy;
    m_stepNumber++;
}

void Sampler::printOutputToTerminal() {
    int     np = m_system->getNumberOfParticles();
    int     nd = m_system->getNumberOfDimensions();
    int     ms = m_system->getNumberOfMetropolisSteps();
    int     p  = m_system->getWaveFunction()->getNumberOfParameters();
    double  ef = m_system->getEquilibrationFraction();
    std::vector<double> pa = m_system->getWaveFunction()->getParameters();

    cout << endl;
    cout << "  -- System info -- " << endl;
    cout << " Number of particles  : " << np << endl;
    cout << " Number of dimensions : " << nd << endl;
    cout << " Number of Metropolis steps run : 10^" << std::log10(ms) << endl;
    cout << " Number of equilibration steps  : 10^" << std::log10(std::round(ms*ef)) << endl;
    cout << endl;
    cout << "  -- Wave function parameters -- " << endl;
    cout << " Number of parameters : " << p << endl;
    for (int i=0; i < p; i++) {
        cout << " Parameter " << i+1 << " : " << pa.at(i) << endl;
    }
    cout << endl;
    cout << "  -- Reults -- " << endl;
    cout << " Energy : " << m_energy << endl;
    cout << " Variance : " << m_variance << endl;
    cout << endl;
}

void Sampler::computeAverages() {
    /* Compute the averages of the sampled quantities. You need to think
     * thoroughly through what is written here currently; is this correct?
     */
    int Dim = m_system->getNumberOfDimensions(); // The Dimension
    int N = m_system->getNumberOfParticles(); // Number of Particles
    int MC = m_system->getNumberOfMetropolisSteps(); // Number of Monte Carlo steps
    m_energy = m_cumulativeEnergy / (MC);
    m_cumulativeEnergy2 = m_cumulativeEnergy2 / (MC);
    m_cumulativeEnergy = m_cumulativeEnergy / (MC);
    m_variance = m_cumulativeEnergy2 - m_cumulativeEnergy*m_cumulativeEnergy;
    m_STD = sqrt(m_variance/(MC));
}


void Sampler::WriteResultstoFile(ofstream& ofile, int MCcycles)
{
  int N = m_system->getNumberOfParticles(); // Number of Particles
  double norm = 1.0/((double) (MCcycles));  // divided by  number of cycles

  double Energy = m_cumulativeEnergy * norm;
  double CumulativeEnergy2 = m_cumulativeEnergy2 *norm;
  double CumulativeEnergy = m_cumulativeEnergy *norm;
  double Variance = CumulativeEnergy2 - CumulativeEnergy*CumulativeEnergy;
  double STD = sqrt(Variance*norm);


  ofile << setiosflags(ios::showpoint | ios::uppercase);
  //ofile << "| Temperature | Energy-Mean | Magnetization-Mean|    Cv    | Susceptibility |\n";


  ofile << "\n";
  //ofile << setw(15) << setprecision(8) << MCcycles; // # Monte Carlo cycles (sweeps per lattice)
  ofile << setw(15) << setprecision(8) << Energy; // Mean energy
  //ofile << setw(15) << setprecision(8) << Variance; // Variance
  //ofile << setw(15) << setprecision(8) << STD; // # Standard deviation


} // end output function
