#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <string>
#include <cmath>
#include <armadillo>
#include <math.h>
#include <cassert>
#include <omp.h>
#include <time.h>
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
using namespace arma;


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

if (Task == "a")
{
  int numberOfSteps       = 1e6;
  int numberOfParticles   = 1;
  int numberOfDimensions  = 3;

  double omega            = 1.0;          // Oscillator frequency.
  double beta             = 1.0;          // Variational parameter.
  double gamma            = 1.0;          // Variational parameter.
  double a                = 0.0;          // Interaction parameter.
  double stepLength       = 1.5;          // Metropolis step length.
  double stepSize         = 1e-2;         // Stepsize in the numerical derivative for kinetic energy
  double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
  double equilibration    = 0.1;          // Amount of the total steps used
  // for equilibration.




  // Analyitcal Run
  cout << "-------------- \n" << "Variational Parameter alpha \n" << "-------------- \n" << endl;

  int Maxiterations = 10;
  vec alphas(Maxiterations); // Variational parameters.
  for (int i = 0; i < Maxiterations; i++){
    alphas(i) = 0.1 + 0.1*i;
  }

  std::vector<double> vecEnergy = std::vector<double>();
  std::vector<double> vecSTD = std::vector<double>();


  for (int i = 0; i < Maxiterations; i++){

    System* system = new System();
    system->setHamiltonian              (new HarmonicOscillator(system, omega));
    system->setWaveFunction             (new SimpleGaussian(system, alphas(i), beta, gamma, a));
    system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
    system->setStepLength               (stepLength);
    system->setPrintOutToTerminal       (false);
    system->runMetropolisSteps          (ofile, numberOfSteps);

    vecEnergy.push_back(system->getSampler()->getEnergy());
    vecSTD.push_back(system->getSampler()->getSTD());

  }

  string file = "Python/Results/Task_a/Variational_parameter.dat";
  ofile.open(file);
  ofile << setiosflags(ios::showpoint | ios::uppercase);
  ofile << setw(15) << setprecision(8) << "alpha "; // alpha
  ofile << setw(15) << setprecision(8) << "Energy "; // Mean energy
  ofile << setw(15) << setprecision(8) << "STD " << endl; // STD

  for (int i = 0; i < Maxiterations; i++){
    ofile << setw(15) << setprecision(8) << alphas(i); // alpha
    ofile << setw(15) << setprecision(8) << vecEnergy[i]; // Mean energy
    ofile << setw(15) << setprecision(8) << vecSTD[i] << endl; // STD

  }

  ofile.close();

}





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
  system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles ));
  system->setStepLength               (stepLength);
  system->runMetropolisSteps          (ofile, numberOfSteps);

  double analytical_Energy = system->getSampler()->getEnergy();
  double analytical_VAR = system->getSampler()->getVAR();
  double analytical_Time = system->getSampler()->getTime();


  // Numerical Run
  cout << "-------------- \n" << "Numerical Run \n" << "-------------- \n" << endl;

  system = new System();
  system->setHamiltonian              (new HarmonicOscillator(system, omega));
  system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
  system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles ));
  system->setStepLength               (stepLength);
  system->setStepSize                 (stepSize);
  system->setNumericalDerivative      (true);
  system->runMetropolisSteps          (ofile, numberOfSteps);

  double numerical_Energy = system->getSampler()->getEnergy();
  double numerical_VAR = system->getSampler()->getVAR();
  double numerical_Time = system->getSampler()->getTime();


  string file = "Python/Results/Task_b/" + to_string(numberOfParticles) + "_particles" + "_" + to_string(numberOfDimensions) + "_dim.dat";
  ofile.open(file);
  ofile << setiosflags(ios::showpoint | ios::uppercase);
  ofile << setw(15) << setprecision(8) << "Energy"; // Mean energy
  ofile << setw(15) << setprecision(8) << "Variance"; // Variance
  ofile << setw(15) << setprecision(8) << "Time" << endl; // Time

  ofile << setw(15) << setprecision(8) << analytical_Energy; // Mean energy
  ofile << setw(15) << setprecision(8) << analytical_VAR; // Variance
  ofile << setw(15) << setprecision(8) << analytical_Time << endl; // Time

  ofile << setw(15) << setprecision(8) << numerical_Energy; // Mean energy
  ofile << setw(15) << setprecision(8) << numerical_VAR; // Variance
  ofile << setw(15) << setprecision(8) << numerical_Time << endl; // Time

  ofile.close();

}


  if (Task == "c")
{

  int numberOfSteps;
  int numberOfParticles;
  int numberOfDimensions;

  double omega            = 1.0;          // Oscillator frequency.
  double alpha            = 0.5;          // Variational parameter.
  double beta             = 1.0;          // Variational parameter.
  double gamma            = beta;         // Variational parameter.
  double a                = 0.0;          // Interaction parameter.
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

    int Maxiterations = 20;
    vec timeStep(Maxiterations); // Timestep to be used in Metropolis-Hastings.
    vec stepLength(Maxiterations); // Metropolis step length.
    for (int i = 0; i < Maxiterations; i++){
      timeStep(i) = 0.1 + 0.1*i;
      stepLength(i) = 0.1 + 0.1*i;
    }


    std::vector<double> vecImportanceSampling = std::vector<double>();
    std::vector<double> vecEnergy_IS = std::vector<double>();
    std::vector<double> vecVAR_IS = std::vector<double>();
    std::vector<double> vecTime_IS = std::vector<double>();



    std::vector<double> vecBruteForce = std::vector<double>();
    std::vector<double> vecEnergy_BF = std::vector<double>();
    std::vector<double> vecVAR_BF = std::vector<double>();
    std::vector<double> vecTime_BF = std::vector<double>();


    cout << "-------------- \n" << "Brute Force Metropolis \n" << "-------------- \n" << endl;

    for (int i = 0; i < Maxiterations; i++){

      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setStepLength               (stepLength(i));
      system->runMetropolisSteps          (ofile, numberOfSteps);

      vecBruteForce.push_back(system->getSampler()->getAcceptedStep());
      vecEnergy_BF.push_back(system->getSampler()->getEnergy());
      vecVAR_BF.push_back(system->getSampler()->getVAR());
      vecTime_BF.push_back(system->getSampler()->getTime());

    }

    cout << "-------------- \n" << "Metropolis-Hastings \n" << "-------------- \n" << endl;

    for (int i = 0; i < Maxiterations; i++){

      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setTimeStep                 (timeStep(i));
      system->setDiffusionCoefficient     (diffusionCoefficient);
      system->setImportanceSampling       (true);
      system->runMetropolisSteps          (ofile, numberOfSteps);

      vecImportanceSampling.push_back(system->getSampler()->getAcceptedStep());
      vecEnergy_IS.push_back(system->getSampler()->getEnergy());
      vecVAR_IS.push_back(system->getSampler()->getVAR());
      vecTime_IS.push_back(system->getSampler()->getTime());

    }


    string file = "Python/Results/Task_c/Brute_Force" + to_string(numberOfParticles) + "_particles" + "_" + to_string(numberOfDimensions) + "_dim.dat";
    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "StepLength"; // StepLength
    ofile << setw(15) << setprecision(8) << "Rate"; // acceptedStep
    ofile << setw(15) << setprecision(8) << "Energy"; // Mean energy
    ofile << setw(15) << setprecision(8) << "Variance"; // Variance
    ofile << setw(15) << setprecision(8) << "Time" << endl; // Time


    for (int i = 0; i < Maxiterations; i++){
      ofile << setw(15) << setprecision(8) << stepLength(i); // stepLength
      ofile << setw(15) << setprecision(8) << vecBruteForce[i]; // acceptedStep
      ofile << setw(15) << setprecision(8) << vecEnergy_BF[i]; // Mean energy
      ofile << setw(15) << setprecision(8) << vecVAR_BF[i]; // Variance
      ofile << setw(15) << setprecision(8) << vecTime_BF[i] << endl; // Time

    }

    ofile.close();



    file = "Python/Results/Task_c/Importance_Sampling" + to_string(numberOfParticles) + "_particles" + "_" + to_string(numberOfDimensions) + "_dim.dat";
    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Timestep"; // timeStep
    ofile << setw(15) << setprecision(8) << "Rate"; // acceptedStep
    ofile << setw(15) << setprecision(8) << "Energy"; // Mean energy
    ofile << setw(15) << setprecision(8) << "Variance"; // Variance
    ofile << setw(15) << setprecision(8) << "Time" << endl; // Time


    for (int i = 0; i < Maxiterations; i++){
      ofile << setw(15) << setprecision(8) << timeStep(i); // stepLength
      ofile << setw(15) << setprecision(8) << vecImportanceSampling[i]; // acceptedStep
      ofile << setw(15) << setprecision(8) << vecEnergy_IS[i]; // Mean energy
      ofile << setw(15) << setprecision(8) << vecVAR_IS[i]; // Variance
      ofile << setw(15) << setprecision(8) << vecTime_IS[i] << endl; // Time
    }

    ofile.close();


}


  if (Task == "d")
  {
    int numberOfSteps;
    int numberOfDimensions = 3;

    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.5;          // Variational parameter.
    double beta             = 2.82843;      // Variational parameter.
    double gamma            = beta;          // Variational parameter.
    double a                = 0.0043;          // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double equilibration    = 0.25;          // Amount of the total steps used
    // for equilibration.


  cout << "\n" << "Which parameters do you want to use?: " << endl;

  cout << "\n" << "The number of Monte Carlo cycles in powers of 2: " << endl;
  cout << "\n" << "Useful information: 2^10 = 10^3 & 2^20 = 10^6" << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfSteps;
  numberOfSteps = pow(2, numberOfSteps);

  std::vector<int> numberOfParticles = std::vector<int>();
  numberOfParticles.push_back(2);
  numberOfParticles.push_back(10);

  std::vector<double> vecSTD = std::vector<double>();
  std::vector<double> vecEnergy = std::vector<double>();



  // Analyitcal Run
  cout << "-------------- \n" << "Statistical Analysis \n" << "-------------- \n" << endl;

  for (int i = 0; i < 2; i++)
  {

      string file = "Python/Results/Task_d/Blocking_Importance_Sampling" + to_string(numberOfParticles[i]) + "_particles" + "_" + to_string(numberOfDimensions) + "_dim.dat";
      ofile.open(file);
      ofile << setiosflags(ios::showpoint | ios::uppercase);
      ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles[i]));
      system->setStepLength               (stepLength);
      system->setRepulsivePotential       (true);
      system->runMetropolisSteps          (ofile, numberOfSteps);

      vecSTD.push_back(system->getSampler()->getSTD());
      vecEnergy.push_back(system->getSampler()->getEnergy());


      ofile.close();
    }

    string file = "Python/Results/Task_d/STD_Importance_Sampling.dat";
    ofile.open(file);
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << "Particles"; // numberOfParticles
    ofile << setw(15) << setprecision(8) << "Energy"; // Mean energy
    ofile << setw(15) << setprecision(8) << "STD" << endl; // Variance


    for (int i = 0; i < 2; i++){
      ofile << setw(15) << setprecision(8) << numberOfParticles[i]; // numberOfParticles
      ofile << setw(15) << setprecision(8) << vecEnergy[i]; // Mean energy
      ofile << setw(15) << setprecision(8) << vecSTD[i] << endl; // Variance
    }

    ofile.close();

  }


  if (Task == "e")
{
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.3;          // Variational parameter.
    double beta             = 2.82843;      // Variational parameter.
    double gamma            = beta;         // Variational parameter.
    double a                = 0.0043;       // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double timeStep         = 1.0;          // Timestep to be used in Metropolis-Hastings.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 0.5;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.


    cout << "\n" << "Which parameters do you want to use?: " << endl;

    cout << "\n" << "The number of Monte Carlo cycles in powers of 2: " << endl;
    cout << "\n" << "Useful information: 2^10 = 10^3 & 2^20 = 10^6" << endl;
    cout << "\n" << "Write here " << endl;
    cin >> numberOfSteps;
    numberOfSteps = pow(2, numberOfSteps);

    cout << "\n" << "The number of Particles: " << endl;
    cout << "\n" << "Write here " << endl;
    cin >> numberOfParticles;


    cout << "\n" << "The number of Dimensions: " << endl;
    cout << "\n" << "Write here " << endl;
    cin >> numberOfDimensions;





    // Analyitcal Run
    cout << "-------------- \n" << "Repulsive Interaction \n" << "-------------- \n" << endl;


    int Maxiterations = 5;
    vec alphas(Maxiterations);
    for (int i = 0; i < Maxiterations; i++){
      alphas(i) = alpha + 0.1*i;
    }

    std::vector<double> vecEnergy = std::vector<double>();
    std::vector<double> vecalpha = std::vector<double>();

    mat Energies_alphas = zeros<mat>(Maxiterations, numberOfSteps);

    double start_time, end_time;
    start_time = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Maxiterations; i++){

      System* system = new System();
      system->setHamiltonian              (new HarmonicOscillator(system, omega));
      system->setWaveFunction             (new SimpleGaussian(system, alphas(i), beta, gamma, a));
      system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
      system->setStepLength               (stepLength);
      system->setRepulsivePotential       (true);
      //system->setPrintOutToTerminal       (false);
      system->runMetropolisSteps          (ofile, numberOfSteps);

      vecEnergy.push_back(system->getSampler()->getEnergy());
      vecalpha.push_back(system->getWaveFunction()->getParameters()[0]);

      for (int j = 0; j < numberOfSteps; j++){
          Energies_alphas(i,j) = system->getSampler()->getEnergies()[j];

      }

    }

    for (int i = 0; i < Maxiterations; i++)
    {
      for (int k = 0; k < Maxiterations; k++)
      {

        if (Energies_alphas(i, numberOfSteps-1) == vecEnergy[k])
        {
          string file = "Python/Results/Task_e/GD_" + to_string(numberOfParticles) + "particles_" + to_string(vecalpha[k]) + "alpha.dat";
          ofile.open(file);
          ofile << setiosflags(ios::showpoint | ios::uppercase);
          ofile << setw(15) << setprecision(8) << "Energy" << endl; // Mean energy

          for (int j = 0; j < numberOfSteps; j++)
          {
            ofile << setw(15) << setprecision(8) << Energies_alphas(i,j) << endl; // Mean energy
          }
          ofile.close();
        }
      }

    }


    for (int i = 0; i < Maxiterations; i++){
      cout << "Energy = " << vecEnergy[i] << "  alpha = " << vecalpha[i] << endl;
    }

    end_time = omp_get_wtime();

    cout << "Time : " << end_time - start_time << endl;

}

  if (Task == "f")
{
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha;                           // Variational parameter.
    double beta             = 2.82843;      // Variational parameter.
    double gamma            = beta;         // Variational parameter.
    double a                = 0.0043;       // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double timeStep         = 1.0;          // Timestep to be used in Metropolis-Hastings.
    double stepSize         = 1e-4;         // Stepsize in the numerical derivative for kinetic energy
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.


    cout << "\n" << "Which parameters do you want to use?: " << endl;

    cout << "\n" << "The number of Monte Carlo cycles in powers of 2: " << endl;
    cout << "\n" << "Useful information: 2^10 = 10^3 & 2^20 = 10^6" << endl;
    cout << "\n" << "Write here " << endl;
    cin >> numberOfSteps;
    numberOfSteps = pow(2, numberOfSteps);

    cout << "\n" << "The number of Particles: " << endl;
    cout << "\n" << "Write here " << endl;
    cin >> numberOfParticles;


    cout << "\n" << "The number of Dimensions: " << endl;
    cout << "\n" << "Write here " << endl;
    cin >> numberOfDimensions;


    // Initialize the seed and call the Mersienne algo
    std::random_device rd;
    std::mt19937_64 gen(rd());
    // Set up the uniform distribution for x \in [[0.3, 0.7]
    std::uniform_real_distribution<double> RandomNumberGenerator(0.3,0.7);

    alpha = RandomNumberGenerator(gen);

    // Analyitcal Run
    cout << "-------------- \n" << " Gradient Descent \n" << "-------------- \n" << endl;

    std::vector<double> vecEnergy = std::vector<double>();
    std::vector<double> vecEnergyDer = std::vector<double>();
    std::vector<double> vecalpha = std::vector<double>();
    std::vector<double> vecdiff = std::vector<double>();

    vec Energies(numberOfSteps);

    double tol = 1e-2;
    double diff = 1;
    double learning_rate = 1e-2;
    int Maxiterations = 50;

    double start_time, end_time;
    start_time = omp_get_wtime();
    int j = 0;
    for (int i = 0; i < Maxiterations; i++){
      if (diff > tol)
      {
        System* system = new System();

        system->setHamiltonian              (new HarmonicOscillator(system, omega));
        system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
        system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
        system->setEquilibrationFraction    (equilibration);
        system->setStepLength               (stepLength);
        system->setDiffusionCoefficient     (diffusionCoefficient);
        system->setRepulsivePotential       (true);
        //system->setPrintOutToTerminal       (false);
        system->runMetropolisSteps          (ofile, numberOfSteps);

        vecEnergy.push_back(system->getSampler()->getEnergy());
        vecEnergyDer.push_back(system->getSampler()->getEnergyDer());
        vecalpha.push_back(system->getWaveFunction()->getParameters()[0]);
        vecdiff.push_back(diff);

        for (int k = 0; k < numberOfSteps; k++){
          Energies(k) = system->getSampler()->getEnergies()[k];
        }

        alpha -= learning_rate*vecEnergyDer[i];

        if (i > 0){
          diff = fabs(vecEnergy[i] - vecEnergy[i-1]);
        }
        j++;
      }
      else
      {
        break;
      }

    }

      end_time = omp_get_wtime();

      cout << "Time : " << end_time - start_time << endl;
      cout << "Iterations : " << j << " / " << Maxiterations << endl;



      string file = "Python/Results/Task_f/Blocking_" + to_string(numberOfParticles) + "_particles.dat";;
      ofile.open(file);
      ofile << setiosflags(ios::showpoint | ios::uppercase);
      ofile << setw(15) << setprecision(8) << "Energy" << endl; // Energy

      for (int k = 0; k < numberOfSteps; k++){
        ofile << setw(15) << setprecision(8) << Energies(k) << endl;
      }
      ofile.close();


      file = "Python/Results/Task_f/Gradient_Descent_" + to_string(numberOfParticles) + "_particles.dat";;
      ofile.open(file);
      ofile << setiosflags(ios::showpoint | ios::uppercase);
      ofile << setw(15) << setprecision(8) << "Alpha"; // Variational parameter
      ofile << setw(15) << setprecision(8) << "Energy"; // Energy
      ofile << setw(15) << setprecision(8) << "Derivative" << endl;

      for (int k = 0; k < j; k++)
      {

      ofile << setw(15) << setprecision(8) << vecalpha[k]; // Variational parameter
      ofile << setw(15) << setprecision(8) << vecEnergy[k]; // Energy
      ofile << setw(15) << setprecision(8) << vecEnergyDer[k] << endl; // Derivative
    }
    ofile.close();

}


  if (Task == "g")
{
    int numberOfSteps;
    int numberOfParticles;
    int numberOfDimensions;
    double omega            = 1.0;          // Oscillator frequency.
    double alpha            = 0.5;          // Variational parameter.
    //double beta             = 1.0;          // Variational parameter. Elliptical case
    double beta             = 2.82843;      // Variational parameter. Ideal case
    double gamma            = beta;         // Variational parameter.
    double a                = 0.0043;       // Interaction parameter.
    double stepLength       = 1.0;          // Metropolis step length.
    double timeStep         = 1.0;          // Timestep to be used in Metropolis-Hastings.
    double diffusionCoefficient  = 1.0;     // DiffusionCoefficient.
    double equilibration    = 0.1;          // Amount of the total steps used
    // for equilibration.
    int numberofBins = 12;
    double binStartpoint = 0;
    double binEndpoint = 2;

  cout << "\n" << "Which parameters do you want to use?: " << endl;

  cout << "\n" << "The number of Monte Carlo cycles in powers of 2: " << endl;
  cout << "\n" << "Useful information: 2^10 = 10^3 & 2^20 = 10^6" << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfSteps;
  numberOfSteps = pow(2, numberOfSteps);

  cout << "\n" << "The number of Particles: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfParticles;


  cout << "\n" << "The number of Dimensions: " << endl;
  cout << "\n" << "Write here " << endl;
  cin >> numberOfDimensions;


  // Analyitcal Run
  cout << "-------------- \n" << " Onebody Densities \n" << "-------------- \n" << endl;

  //string file = "Python/Results/Task_g/IdealOnebody_Density.dat";
  //string file = "Python/Results/Task_g/Onebody_Density.dat";
  string file = "Python/Results/Task_g/RepulsiveOnebody_Density.dat";
  //string file = "Python/Results/Task_g/StrongRepulsiveOnebody_Density.dat";
  //string file = "Python/Results/Task_g/StrongestRepulsiveOnebody_Density.dat";


  ofile.open(file);
  ofile << setiosflags(ios::showpoint | ios::uppercase);
  ofile << setw(15) << setprecision(8) << "Bins"; // Variational parameter
  ofile << setw(15) << setprecision(8) << "Counter" << endl; // Variational parameter

  System* system = new System();
  system->setHamiltonian              (new HarmonicOscillator(system, omega));
  system->setWaveFunction             (new SimpleGaussian(system, alpha, beta, gamma, a));
  system->setInitialState             (new RandomUniform(system, numberOfDimensions, numberOfParticles));
  system->setEquilibrationFraction    (equilibration);
  system->setStepLength               (stepLength);
  system->setDiffusionCoefficient     (diffusionCoefficient);
  system->setBinStartpoint            (binStartpoint);
  system->setBinEndpoint              (binEndpoint);
  system->setNumberofBins             (numberofBins);
  system->setBinVector                (binStartpoint, binEndpoint, numberofBins);
  system->setOneBodyDensity           (true);
  system->setRepulsivePotential       (true);
  system->runMetropolisSteps          (ofile, numberOfSteps);

  ofile.close();

}

    return 0;
}
