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
    void setacceptedSteps(int counter);
    void setEnergies(int OptCycles);
    void setBlocking(int MCcycle);
    void initializeVariables();
    void sample();
    void printOutputToTerminal(double time);
    void computeAverages(double time, int numberOfProcesses, int myRank);
    void Blocking(int MCcycle, int numberOfProcesses, int myRank);
    void Energies(int OptCycles, int myRank);
    void WriteBlockingtoFile(ofstream& ofile, int myRank);

    double getSTD()                   { return m_STD; }
    double getVAR()                   { return m_variance; }
    double getEnergy()                { return m_energy; }
    double getEnergyDer()             { return m_EnergyDer; }
    double getTime()                  { return m_totalTime; }
    double getAcceptedSteps()         { return m_globalacceptedSteps; }
    vec getEnergies()                 { return m_Energies; }
    vec getBlocking()                 { return m_Blocking; }



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
    double  m_localacceptedSteps = 0;
    double  m_globalacceptedSteps = 0;

    vec m_Energies;
    vec m_Blocking;

    vec m_aDelta;
    vec m_bDelta;
    vec m_wDelta;

    vec m_EaDelta;
    vec m_EbDelta;
    vec m_EwDelta;

    vec m_agrad;
    vec m_bgrad;
    vec m_wgrad;



    class System* m_system = nullptr;
};
