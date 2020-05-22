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
  cout << "\n" << "Project Task B -  Brute Force: " <<  "Write b " << endl;
  cout << "\n" << "Project Task C -  Importance Sampling: " <<  "Write c " << endl;
  cout << "\n" << "Project Task F  - Gibbs sampling: " <<  "Write f " << endl;



  // Chosen parameters

  // Optimizer parameters
  double eta              = pow(2, -1);   // Learning rate
  int numberOfHidden      = 2;            // Number of hidden units


  // Initialisation parameters
  int numberOfParticles   = 1;
  int numberOfDimensions  = 1;
  int numberOfInputs      = numberOfParticles*numberOfDimensions;  // Number of visible units
  double sigma            = 1.0;          // Normal distribution visibles
  double gibbs            = 1.0;          // Gibbs parameter to change the wavefunction
  bool gaussianInitialization = false; // Weights & biases (a,b,w) initialized uniformly or gaussian


  // Sampler parameters
  int OptCycles           = 500;          // Number of optimization iterations
  int MCcycles            = pow(2, 17);   // Number of samples in each iteration
  double stepLength       = 1.0;         // Metropolis step length.
  double timeStep         = 0.5;         // Timestep to be used in Metropolis-Hastings
  double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.

  // Hamiltonian parameters
  double omega            = 1.0;          // Oscillator frequency.
  bool includeInteraction = false;      // Include interaction or not


  cout << "\n" << "Write here " << endl;
  string Task;
  cin >> Task;

  // Parameter for files
  int gamma = log2(eta);
  int mc = log2(MCcycles);

  string file;
  //Benchmark task a.
  if (Task == "b"){

    // Analytical Run
    cout << "-------------- \n" << "Brute Force \n" << "-------------- \n" << endl;

    // Choose which file to write to
    //string file = "Python/Results/Statistical_Analysis/BF_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //string file = "Python/Results/Statistical_Analysis/I_BF_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setStepLength               (stepLength);
    system->setEquilibrationFraction    (equilibration);

    system->setRepulsivePotential       (includeInteraction);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();


    // Write to file
    //file = "Python/Results/Brute_Force/E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Brute_Force/I_E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycles
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    for (int i = 0; i < OptCycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Iteration
      ofile << setw(15) << setprecision(8) << system->getSampler()->getEnergies()(i) << endl; // Mean energy

    }

    ofile.close();

  }

  if (Task == "c"){

    cout << "-------------- \n" << "Importance Sampling \n" << "-------------- \n" << endl;

    // Choose which file to write to
    //string file = "Python/Results/Statistical_Analysis/IS_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //string file = "Python/Results/Statistical_Analysis/I_IS_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setTimeStep                 (timeStep);
    system->setDiffusionCoefficient     (diffusionCoefficient);
    system->setEquilibrationFraction    (equilibration);
    system->setImportanceSampling       (true);

    system->setRepulsivePotential       (includeInteraction);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();


    // Write to file
    //file = "Python/Results/Importance_Sampling/E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Importance_Sampling/I_E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycles
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    for (int i = 0; i < OptCycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Iteration
      ofile << setw(15) << setprecision(8) << system->getSampler()->getEnergies()(i) << endl; // Mean energy

    }

    ofile.close();

  }



  if (Task == "f"){

    // Analytical Run
    cout << "-------------- \n" << "Gibbs sampling \n" << "-------------- \n" << endl;
    gibbs = 2;

    // Choose which file to write to
    //string file = "Python/Results/Statistical_Analysis/GI_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //string file = "Python/Results/Statistical_Analysis/I_GI_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    //Initialise the system.
    System* system = new System();
    system->setNetwork                  (new NeuralNetwork(system, eta, numberOfInputs, numberOfHidden));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles, numberOfHidden, gaussianInitialization));
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new NeuralQuantumState(system, sigma, gibbs));
    system->setGibbsSampling            (true);
    system->setEquilibrationFraction    (equilibration);

    system->setRepulsivePotential       (includeInteraction);
    system->setPrintOutToTerminal       (true);
    system->runOptimizer                (ofile, OptCycles, MCcycles);

    ofile.close();

    // Write to file
    //file = "Python/Results/Gibbs/E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";
    //file = "Python/Results/Gibbs/I_E_eta_2^" + to_string(gamma) + "_nh_" + to_string(numberOfHidden) + "_nx_" + to_string(numberOfInputs) + "MC_2^" + to_string(mc) + ".dat";

    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Iteration"; // OptCycles
    ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

    for (int i = 0; i < OptCycles; i++){
      ofile << setw(15) << setprecision(8) << i+1; // Iteration
      ofile << setw(15) << setprecision(8) << system->getSampler()->getEnergies()(i) << endl; // Mean energy

    }

    ofile.close();

  }


  return 0;
}
