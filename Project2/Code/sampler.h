#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

#include "Eigen"
using namespace std;

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void setEnergies(int MCcycles);
    void setGradients();
    void sample(bool acceptedStep, int MCcycles);
    void printOutputToTerminal(double time, double acceptedStep);
    void computeAverages(double time, double acceptedStep);

    void WriteResultstoFile(ofstream& ofile, int MCcycles);
    double getSTD()                   { return m_STD; }
    double getVAR()                   { return m_variance; }
    double getEnergy()                { return m_energy; }
    double getEnergyDer()             { return m_EnergyDer; }
    double getTime()                  { return m_totalTime; }
    double getAcceptedStep()          { return m_acceptedStep; }
    vector<double> getEnergies()      { return m_Energies; }



private:
    int     m_numberOfMetropolisSteps = 0;
    int     m_stepNumber = 0;
    double  m_energy = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy2 = 0;
    double  m_DeltaPsi  = 0;
    double  m_DerivativePsiE  = 0;
    double  m_EnergyDer  = 0;
    double  m_variance = 0;
    double  m_STD = 0;
    double  m_totalTime = 0;
    double  m_acceptedStep = 0;

    std::vector<double>  m_Energies;

    Eigen::VectorXd  m_aDelta;
    Eigen::VectorXd  m_bDelta;
    Eigen::VectorXd  m_wDelta;

    Eigen::VectorXd  m_EaDelta;
    Eigen::VectorXd  m_EbDelta;
    Eigen::VectorXd  m_EwDelta;

    Eigen::VectorXd  m_agrad;
    Eigen::VectorXd  m_bgrad;
    Eigen::VectorXd  m_wgrad;


    class System* m_system = nullptr;
};
