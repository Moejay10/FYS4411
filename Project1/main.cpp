#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include "system.h"
#include "particle.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/simplegaussian.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "Math/random.h"
ofstream ofile;
using namespace std;


int main() {

  int numberOfDimensions;
  int numberOfParticles;
  int numberOfSteps       = (int) 1e4;
  double omega            = 1.0;          // Oscillator frequency.
  double alpha            = 0.5;          // Variational parameter.
  double stepLength       = 1.0;          // Metropolis step length.
  double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.



cout << "\n" << "Which parameters do you want to use?: " << endl;

cout << "\n" << "The number of Particles: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfParticles;


cout << "\n" << "The number of Dimensions: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfDimensions;



/*
    int numberOfDimensions  = 1;
    int numberOfParticles   = 1;
    int numberOfSteps       = (int) 1e6;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.5;          // Variational parameter.
    double stepLength       = 0.1;          // Metropolis step length.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.
*/
    string file = "Python/Results/Energies.dat";
    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "MCcycles"; // # Monte Carlo cycles (sweeps per lattice)
    ofile << setw(15) << setprecision(8) << "Energy"; // Mean energy
    ofile << setw(15) << setprecision(8) << "Variance"; // Variance
    ofile << setw(15) << setprecision(8) << "STD"; // # Standard deviation

    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->runMetropolisSteps          (ofile, numberOfSteps);

    ofile.close();
    return 0;
}
