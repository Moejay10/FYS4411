#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
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
  double stepLength       = 1.0;          // Metropolis step length.
  double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
  double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.
  bool numerical_derivative;
  bool brute_force = true;

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

    // Analyitcal Run
    cout << "-------------- \n" << "Analyitcal Run \n" << "-------------- \n" << endl;
    numerical_derivative = false;
    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setStepSize                 (stepSize);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->runMetropolisSteps          (ofile, numerical_derivative, brute_force, numberOfSteps);

    // Numerical Run
    cout << "-------------- \n" << "Numerical Run \n" << "-------------- \n" << endl;
    numerical_derivative = true;
    system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setStepSize                 (stepSize);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->runMetropolisSteps          (ofile, numerical_derivative, brute_force, numberOfSteps);

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
  double stepLength       = 1.0;          // Metropolis step length.
  double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
  double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.
  bool numerical_derivative = false;
  bool brute_force = false;

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

    // Analyitcal Run
    // Importance Sampling
    cout << "-------------- \n" << "Importance Sampling \n" << "-------------- \n" << endl;

    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setEquilibrationFraction    (equilibration);
    system->setStepLength               (stepLength);
    system->setStepSize                 (stepSize);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->runMetropolisSteps          (ofile, numerical_derivative, brute_force, numberOfSteps);

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
    double stepLength       = 1.0;          // Metropolis step length.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.
    bool numerical_derivative = false;
    bool brute_force = false;

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
  //ofile << setw(15) << setprecision(8) << "MCcycles"; // # Monte Carlo cycles (sweeps per lattice)
  ofile << setw(15) << setprecision(8) << "Energy"; // Mean energy
  //ofile << setw(15) << setprecision(8) << "Variance"; // Variance
  //ofile << setw(15) << setprecision(8) << "STD"; // # Standard deviation


      // Analyitcal Run
      cout << "-------------- \n" << "Statistical Analysis \n" << "-------------- \n" << endl;
      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setEquilibrationFraction    (equilibration);
      system->setStepLength               (stepLength);
      system->setStepSize                 (stepSize);
      system->setDiffusionCoefficient     (diffusionCoefficient);
      system->runMetropolisSteps          (ofile, numerical_derivative, brute_force, numberOfSteps);

      ofile.close();
  }


  if (Task == "e")
  {
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.5;          // Variational parameter.
    double beta             = 1.0;          // Variational parameter.
    double gamma            = 1.0;          // Variational parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.
    bool numerical_derivative = false;
    bool brute_force = false;

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




      // Analyitcal Run
      cout << "-------------- \n" << "Repulsive Interaction \n" << "-------------- \n" << endl;
      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setEquilibrationFraction    (equilibration);
      system->setStepLength               (stepLength);
      system->setStepSize                 (stepSize);
      system->setDiffusionCoefficient     (diffusionCoefficient);
      system->setRepulsivePotential       (true);
      system->runMetropolisSteps          (ofile, numerical_derivative, brute_force, numberOfSteps);

  }

  if (Task == "f")
  {
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.1;          // Variational parameter.
    double beta             = 2.82843;          // Variational parameter.
    double gamma            = beta;          // Variational parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.
    bool numerical_derivative = false;
    bool brute_force = false;

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
        system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma));
        system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
        system->setEquilibrationFraction    (equilibration);
        system->setStepLength               (stepLength);
        system->setStepSize                 (stepSize);
        system->setDiffusionCoefficient     (diffusionCoefficient);
        system->setRepulsivePotential       (true);
        system->runMetropolisSteps          (ofile, numerical_derivative, brute_force, numberOfSteps);

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
