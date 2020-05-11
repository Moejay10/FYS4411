#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <armadillo>

using namespace arma;
using namespace std;

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void setMCcyles(int effectiveSamplings);
    void setacceptedStep(int counter);
    void setEnergies(int MCcycles);
    void setGradients();
    void sample();
    void printOutputToTerminal(double time);
    void computeAverages(double time, int numberOfProcesses, int myRank);
    void Analysis(int MCcycles, int numberOfProcesses, int myRank);
    void WriteBlockingtoFile(ofstream& ofile, int MCcycles);

    double getSTD()                   { return m_STD; }
    double getVAR()                   { return m_variance; }
    double getEnergy()                { return m_energy; }
    double getEnergyDer()             { return m_EnergyDer; }
    double getTime()                  { return m_totalTime; }
    double getAcceptedStep()          { return m_acceptedStep; }
    vec getEnergies()                 { return m_Energies; }



private:
    int     m_numberOfMetropolisSteps = 0;
    int     m_MCcycles = 0;
    int     m_stepNumber = 0;
    double  m_energy = 0;
    double  m_localenergy = 0;
    double  m_globalenergy = 0;
    double  m_localcumulativeEnergy = 0;
    double  m_globalcumulativeEnergy = 0;
    double  m_localcumulativeEnergy2 = 0;
    double  m_globalcumulativeEnergy2 = 0;
    double  m_DeltaPsi  = 0;
    double  m_DerivativePsiE  = 0;
    double  m_EnergyDer  = 0;
    double  m_variance = 0;
    double  m_STD = 0;
    double  m_totalTime = 0;
    double  m_acceptedStep = 0;

    vec m_Energies;

    vec m_localaDelta;
    vec m_localbDelta;
    vec m_localwDelta;

    vec m_localEaDelta;
    vec m_localEbDelta;
    vec m_localEwDelta;

    vec m_agrad;
    vec m_bgrad;
    vec m_wgrad;

    vec m_globalaDelta;
    vec m_globalbDelta;
    vec m_globalwDelta;

    vec m_globalEaDelta;
    vec m_globalEbDelta;
    vec m_globalEwDelta;;

    class System* m_system = nullptr;
};
