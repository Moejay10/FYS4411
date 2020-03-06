#pragma once
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
using namespace std;

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void sample(bool acceptedStep, int MCcycles);
    void printOutputToTerminal(double time, double acceptedStep);
    void computeAverages();
    void Probability();
    void WriteOneBodyDensitytoFile(ofstream& ofile);

    void WriteResultstoFile(ofstream& ofile, int MCcycles);
    double getEnergy()          { return m_energy; }
    double getEnergyDer()       { return m_EnergyDer; }


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
    class System* m_system = nullptr;
};
