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
#include <omp.h>

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
using namespace arma;
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
//     system->runMetropolisSteps          (ofile, MCcycles);
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
  cout << "\n" << "Project Task B -  Initial Code: " <<  "Write b " << endl;
  cout << "\n" << "Project Task C -  Importance Sampling: " <<  "Write c " << endl;
  cout << "\n" << "Project Task D -  Statistical Analysis: " <<  "Write d " << endl;
  cout << "\n" << "Project Task F  - Gibbs sampling: " <<  "Write e " << endl;
  cout << "\n" << "Project Task G  - Interaction: " <<  "Write g " << endl;



  // Chosen parameters
  int OptCycles           = 50;
  int MCcycles            = 1e4;
  int numberOfParticles   = 2;
  int numberOfDimensions  = 2;
  int numberOfHidden      = 2;            // Number of hidden units
  double sigma            = 1.0;          // Normal distribution visibles
  double gibbs            = 2.0;          // Gibbs parameter to change the wavefunction
  double eta              = 0.01;         // Learning rate
  double omega            = 1.0;          // Oscillator frequency.
  double stepLength       = 0.45;         // Metropolis step length.
  double timeStep         = 1.0;          // Timestep to be used in Metropolis-Hastings
  double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.


  cout << "\n" << "Write here " << endl;
  string Task;
  cin >> Task;

  //Benchmark task a.
  if (Task == "b"){

    // Analytical Run
    cout << "-------------- \n" << "Initial Code \n" << "-------------- \n" << endl;

      //Initialise the system.
      System* system = new System();
      system->setNetwork                  (new NeuralNetwork(system, eta));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden));
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
      system->setStepLength               (stepLength);
      system->setPrintOutToTerminal       (true);
      system->runOptimizer                (ofile, OptCycles, MCcycles);

  }

  if (Task == "c"){

    cout << "-------------- \n" << "Importance Sampling \n" << "-------------- \n" << endl;

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setTimeStep                 (timeStep);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->setImportanceSampling       (true);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

  }


  if (Task == "d"){

    cout << "-------------- \n" << "Statistical Analysis \n" << "-------------- \n" << endl;

    MCcycles = 15;
    MCcycles = pow(2, MCcycles);

    // Choose which file to write to
    string file = "Python/Results/Task_d/Blocking.dat";
    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setTimeStep                 (timeStep);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->setImportanceSampling       (true);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();

  }


  if (Task == "f"){

    // Analytical Run
    cout << "-------------- \n" << "Gibbs sampling \n" << "-------------- \n" << endl;
    gibbs = 2;

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setGibbsSampling            (true);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

  }


  if (Task == "g"){

    // Analytical Run
    cout << "-------------- \n" << "Interaction \n" << "-------------- \n" << endl;

      //Initialise the system.
      System* system = new System();
      system->setNetwork                  (new NeuralNetwork(system, eta));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden));
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
      system->setStepLength               (stepLength);
      system->setRepulsivePotential       (true);
      system->setPrintOutToTerminal       (true);
      system->runOptimizer                (ofile, OptCycles, MCcycles);

  }




  return 0;
}
