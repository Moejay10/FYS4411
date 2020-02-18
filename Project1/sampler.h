#pragma once
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace std;

class Sampler {
public:
    Sampler(class System* system);
    void setNumberOfMetropolisSteps(int steps);
    void sample(ofstream& ofile, bool acceptedStep, int MCcycles);
    void printOutputToTerminal();
    void computeAverages();
    void WriteResultstoFile(ofstream& ofile, int MCcycles);
    double getEnergy()          { return m_energy; }

private:
    int     m_numberOfMetropolisSteps = 0;
    int     m_stepNumber = 0;
    double  m_energy = 0;
    double  m_cumulativeEnergy = 0;
    double  m_cumulativeEnergy2 = 0;
    double  m_variance = 0;
    double  m_STD = 0;
    class System* m_system = nullptr;
};
