#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <math.h>
#include <cassert>
#include "system.h"
#include "particle.h"
#include "sampler.h"
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

cout << "\n" << "Which Project Task do you want to run?: " << endl;
cout << "\n" << "Project Task B - Analytical vs Numerical: " <<  "Write b " << endl;
cout << "\n" << "Project Task C - Importance Sampling: " <<  "Write c " << endl;
cout << "\n" << "Project Task D - Statistical Analysis: " <<  "Write d " << endl;
cout << "\n" << "Project Task E  - Repulsive Interaction: " <<  "Write e " << endl;
cout << "\n" << "Project Task F  - Gradient Descent: " <<  "Write f " << endl;
//cout << "\n" << "Project Task G  - Onebody Densities: " <<  "Write g " << endl;


cout << "\n" << "Write here " << endl;
string Task;
cin >> Task;

if (Task == "b")
  {
  int numberOfSteps;
  int numberOfParticles;
  int numberOfDimensions;
  double omega            = 1.0;          // Oscillator frequency.
  double alpha            = 0.5;          // Variational parameter.
  double beta             = 1.0;          // Variational parameter.
  double gamma            = 1.0;          // Variational parameter.
  double a                = 0.0;          // Interaction parameter.
  double stepLength       = 1.0;          // Metropolis step length.
  double stepSize         = 1e-2;         // Stepsize in the numerical derivative for kinetic energy
  double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.


cout << "\n" << "Which parameters do you want to use?: " << endl;

cout << "\n" << "The number of Monte Carlo cycles: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfSteps;

cout << "\n" << "The number of Particles: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfParticles;


cout << "\n" << "The number of Dimensions: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfDimensions;





    // Analyitcal Run
    cout << "-------------- \n" << "Analyitcal Run \n" << "-------------- \n" << endl;
    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setStepSize                 (stepSize);
    system->runMetropolisSteps          (ofile, numberOfSteps);

    // Numerical Run
    cout << "-------------- \n" << "Numerical Run \n" << "-------------- \n" << endl;
    system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepSize                 (stepSize);
    system->setStepLength               (stepLength);
    system->setNumericalDerivative      (true);
    system->runMetropolisSteps          (ofile, numberOfSteps);

  }


  if (Task == "c")
  {

  int numberOfSteps;
  int numberOfParticles;
  int numberOfDimensions;
  double omega            = 1.0;          // Oscillator frequency.
  double alpha            = 0.5;          // Variational parameter.
  double beta             = 1.0;          // Variational parameter.
  double gamma            = 1.0;          // Variational parameter.
  double a                = 0.0;          // Interaction parameter.
  double stepLength       = 1.0;          // Metropolis step length.
  double timeStep         = 1.0;          // Timestep to be used in Metropolis-Hastings.
  double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
  double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.


cout << "\n" << "Which parameters do you want to use?: " << endl;

cout << "\n" << "The number of Monte Carlo cycles: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfSteps;

cout << "\n" << "The number of Particles: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfParticles;


cout << "\n" << "The number of Dimensions: " << endl;
cout << "\n" << "Write here " << endl;
cin >> numberOfDimensions;



    // Analyitcal Run
    // Importance Sampling
    cout << "-------------- \n" << "Importance Sampling \n" << "-------------- \n" << endl;

    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setTimeStep                 (timeStep);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->setImportanceSampling       (true);
    system->runMetropolisSteps          (ofile, numberOfSteps);

  }


  if (Task == "d")
  {
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.5;          // Variational parameter.
    double beta             = 1.0;          // Variational parameter.
    double gamma            = 1.0;          // Variational parameter.
    double a                = 0.0;          // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.


  cout << "\n" << "Which parameters do you want to use?: " << endl;

  cout << "\n" << "The number of Monte Carlo cycles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfSteps;
  double x = log2(numberOfSteps);
  int y = (int) x;
  double diff = x-y;
  cout << "Warning: log2(MCcycles) must be an integer" << endl;


  cout << "\n" << "The number of Particles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfParticles;


  cout << "\n" << "The number of Dimensions: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfDimensions;


  string file = "Python/Results/Energies.dat";
  ofile.open(file);
  //ofile << setiosflags(ios::showpoint | ios::uppercase);
  //ofile << setw(15) << setprecision(8) << "MCcycles"; // # Monte Carlo cycles (sweeps per lattice)
  ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy
  //ofile << setw(15) << setprecision(8) << "Variance"; // Variance
  //ofile << setw(15) << setprecision(8) << "STD"; // # Standard deviation


      // Analyitcal Run
      cout << "-------------- \n" << "Statistical Analysis \n" << "-------------- \n" << endl;
      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setEquilibrationFraction    (equilibration);
      system->setStepLength               (stepLength);
      system->setDiffusionCoefficient     (diffusionCoefficient);
      system->setImportanceSampling       (true);
      system->runMetropolisSteps          (ofile, numberOfSteps);

      ofile.close();
  }


  if (Task == "e")
  {
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.5;          // Variational parameter.
    double beta             = 2.82843;      // Variational parameter.
    double gamma            = beta;         // Variational parameter.
    double a                = 0.0043;       // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.


  cout << "\n" << "Which parameters do you want to use?: " << endl;

  cout << "\n" << "The number of Monte Carlo cycles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfSteps;

  cout << "\n" << "The number of Particles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfParticles;


  cout << "\n" << "The number of Dimensions: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfDimensions;


  string file = "Exercise_e.dat";
  ofile.open(file);
  ofile << setiosflags(ios::showpoint | ios::uppercase);
  ofile << setw(15) << setprecision(8) << "Energy "; // Mean energy
  ofile << setw(15) << setprecision(8) << "  CumulativeEnergy" << endl;; // # Monte Carlo cycles (sweeps per lattice)


      // Analyitcal Run
      cout << "-------------- \n" << "Repulsive Interaction \n" << "-------------- \n" << endl;
      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setEquilibrationFraction    (equilibration);
      system->setDiffusionCoefficient     (diffusionCoefficient);
      system->setStepLength               (stepLength);
      system->setRepulsivePotential       (true);
      //system->setImportanceSampling       (true);
      system->runMetropolisSteps          (ofile, numberOfSteps);

      ofile.close();

  }

  if (Task == "f")
  {
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.1;          // Variational parameter.
    double beta             = 2.82843;      // Variational parameter.
    double gamma            = beta;         // Variational parameter.
    double a                = 0.0043;       // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.


  cout << "\n" << "Which parameters do you want to use?: " << endl;

  cout << "\n" << "The number of Monte Carlo cycles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfSteps;

  cout << "\n" << "The number of Particles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfParticles;


  cout << "\n" << "The number of Dimensions: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfDimensions;



      // Analyitcal Run
      cout << "-------------- \n" << " Gradient Descent \n" << "-------------- \n" << endl;

      std::vector<double> vecEnergy = std::vector<double>();
      std::vector<double> vecEnergyDer = std::vector<double>();

      double tol = 1e-8;
      double diff = 0;
      double learning_rate = 1e-0;
      int Niterations = 50;

      for (int i = 0; i < Niterations; i++){
        System* system = new System();

        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
        system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
        system->setEquilibrationFraction    (equilibration);
        system->setStepLength               (stepLength);
        system->setDiffusionCoefficient     (diffusionCoefficient);
        system->setRepulsivePotential       (true);
        system->runMetropolisSteps          (ofile, numberOfSteps);

        vecEnergy.push_back(system->getSampler()->getEnergy());
        vecEnergyDer.push_back(system->getSampler()->getEnergyDer());

        if (i > 0){
          diff = vecEnergy[i] - vecEnergy[i-1];
        }
        alpha -= learning_rate*vecEnergyDer[i];
        if (diff < tol){
          cout << "best alpha was obtained at " << alpha << endl;
          break;
        }
      }
  }

    return 0;
}
