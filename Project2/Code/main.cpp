// Dependencies
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <math.h>
#include <cassert>
#include <time.h>
#include <armadillo>

// Include all header files
#include "system.h"
#include "sampler.h"
#include "NeuralNetworks/network.h"
#include "NeuralNetworks/neuralnetwork.h"
#include "WaveFunctions/wavefunction.h"
#include "WaveFunctions/neuralquantumstate.h"
#include "Hamiltonians/hamiltonian.h"
#include "Hamiltonians/harmonicoscillator.h"
#include "InitialStates/initialstate.h"
#include "InitialStates/randomuniform.h"
#include "Math/random.h"

ofstream ofile;
using namespace std;

// #############################################################################
// ############################### Description: ################################
// #############################################################################
// This is the main program where one chooses the constants, initialise the
// system, and eventually adjust the variational parameter alpha as well.
// This file is heavily dependent on what every task in the project description
// explicitly asks for, however, it is quite easy to add a more general program
// for a normal Monte Carlo run with your chosen parameters.
//
// #############################################################################
// ########################## Optional parameters: #############################
// #############################################################################
//    double omega            = 1.0;          // Oscillator frequency.
//    double alpha            = 0.5;          // Variational parameter.
//    double beta             = 1.0;          // Variational parameter. Elliptical case
//    double beta             = 2.82843;      // Variational parameter. Ideal case
//    double gamma            = beta;         // Variational parameter.
//    double a                = 0.0043;       // Interaction parameter.
//    double stepLength       = 1.0;          // Metropolis step length.
//    double timeStep         = 1.0;          // Timestep to be used in Metropolis-Hastings.
//    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
//    double equilibration    = 0.1;          // Amount of the total steps used for equilibration.
//
//    int numberofBins = 20;                  // Number of bins for OneBodyDensity
//    double binStartpoint = 0;               // Where to start to count bins
//    double binEndpoint = 2;                 // Where to end the counting of bins
//
// #############################################################################
// ############# How to initialise a system and its parameters #################
// #############################################################################
//   - Some parameters have to be initialised for a system to run, while others
//     are not needed.
//     The ones that HAS to be initialised is:
//
// #############################################################################
//     System* system = new System();
//     system->setHamiltonian              (new HarmonicOscillator(system, omega));
//     system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
//     system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
//     system->setEquilibrationFraction    (equilibration);
//     system->setStepLength               (stepLength);
//     system->setDiffusionCoefficient     (diffusionCoefficient);
//     system->setPrintOutToTerminal       (true); // true or false
//     system->runMetropolisSteps          (ofile, numberOfSteps);
// #############################################################################
//  - The optional initialiser:
// #############################################################################
//     system->setRepulsivePotential       (true); // true or false
//     system->setNumericalDerivative      (true); // true or false
//    system->setImportanceSampling       (true);  // true or false
//    system->setTimeStep                 (timeStep(i)); // set if setImportanceSampling == true
//
//    system->setOneBodyDensity           (true); // true or false
//    system->setBinStartpoint            (binStartpoint);
//    system->setBinEndpoint              (binEndpoint);
//    system->setNumberofBins             (numberofBins);
//    system->setBinVector                (binStartpoint, binEndpoint, numberofBins);
// #############################################################################


int main() {

  cout << "\n" << "Which Project Task do you want to run?: " << endl;
  cout << "\n" << "Project Task A - Variational Parameter Alpha: " <<  "Write a " << endl;
  cout << "\n" << "Project Task B - Analytical vs Numerical: " <<  "Write b " << endl;
  cout << "\n" << "Project Task C - Importance Sampling: " <<  "Write c " << endl;
  cout << "\n" << "Project Task D - Statistical Analysis: " <<  "Write d " << endl;
  cout << "\n" << "Project Task E  - Repulsive Interaction: " <<  "Write e " << endl;
  cout << "\n" << "Project Task F  - Gradient Descent: " <<  "Write f " << endl;
  cout << "\n" << "Project Task G  - Onebody Densities: " <<  "Write g " << endl;


  cout << "\n" << "Write here " << endl;
  string Task;
  cin >> Task;

  //Benchmark task a.
  if (Task == "a"){

      // Chosen parameters
      int OptCycles           = 10;
      int MCcycles            = 1e5;
      int numberOfParticles   = 2;
      int numberOfDimensions  = 2;
      int numberOfHidden        = 2;          // Number of hidden units
      double sigma            = 1.0;          // Normal distribution visibles
      double eta              = 0.01;         // Learning rate
      double omega            = 1.0;          // Oscillator frequency.
      double stepLength       = 1.5;          // Metropolis step length.
      double stepSize         = 1e-2;         // Stepsize in the numerical derivative for kinetic energy
      double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
      double equilibration    = 0.1;          // Amount of the total steps used
      // for equilibration.

      // Analytical Run
      cout << "-------------- \n" << "Variational Parameter alpha \n" << "-------------- \n" << endl;

        //Initialise the system.
        System* system = new System();
        system->setNetwork                  (new NeuralNetwork(system, eta));

        system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden));

        system->setHamiltonian              (new HarmonicOscillator(system, omega));

        system->setWaveFunction             (new NeuralQuantumState(system, sigma));

        system->setStepLength               (stepLength);
        system->setPrintOutToTerminal       (true);

        system->runOptimizer                (ofile, OptCycles, MCcycles);

    }



  return 0;
}
