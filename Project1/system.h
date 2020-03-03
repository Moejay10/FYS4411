#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <time.h>
#include <omp.h>
using namespace std;

class System {
public:
    bool ImportanceMetropolisStep   (int);
    bool metropolisStep             (int);
    bool setRepulsivePotential      (bool);
    bool setImportanceSampling      (bool);
    bool setNumericalDerivative     (bool);
    void runMetropolisSteps         (ofstream& ofile, int numberOfMetropolisSteps);
    void setNumberOfParticles       (int numberOfParticles);
    void setNumberOfDimensions      (int numberOfDimensions);
    void setStepLength              (double stepLength);
    void setTimeStep                (double timeStep);
    void setStepSize                (double stepSize);
    void setEquilibrationFraction   (double equilibrationFraction);
    void setDiffusionCoefficient    (double diffusionCoefficient);
    void setHamiltonian             (class Hamiltonian* hamiltonian);
    void setWaveFunction            (class WaveFunction* waveFunction);
    void setInitialState            (class InitialState* initialState);
    class WaveFunction*             getWaveFunction()   { return m_waveFunction; }
    class Hamiltonian*              getHamiltonian()    { return m_hamiltonian; }
    class Sampler*                  getSampler()        { return m_sampler; }
    std::vector<class Particle*>    getParticles()      { return m_particles; }
    int getNumberOfParticles()          { return m_numberOfParticles; }
    int getNumberOfDimensions()         { return m_numberOfDimensions; }
    int getNumberOfMetropolisSteps()    { return m_numberOfMetropolisSteps; }
    double getEquilibrationFraction()   { return m_equilibrationFraction; }
    double getStepLength()              { return m_stepLength; }
    double getTimeStep()                { return m_timeStep; }
    double getStepSize()                { return m_stepSize; }
    double getDiffusionCoefficient()    { return m_diffusionCoefficient; }
    bool   getRepulsivePotential()      { return m_statement;}
    bool   getImportanceSampling()      { return m_importance_sampling;}
    bool   getNumericalDerivative()     { return m_numerical_dericative;}


private:
    bool                            m_statement = false;
    bool                            m_importance_sampling = false;
    bool                            m_numerical_dericative = false;
    int                             m_numberOfParticles = 0;
    int                             m_numberOfDimensions = 0;
    int                             m_numberOfMetropolisSteps = 0;
    double                          m_equilibrationFraction = 0.0;
    double                          m_stepLength = 0.1;
    double                          m_timeStep = 0.01;
    double                          m_stepSize = 0.1;
    double                          m_diffusionCoefficient = 1.0;
    class WaveFunction*             m_waveFunction = nullptr;
    class Hamiltonian*              m_hamiltonian = nullptr;
    class InitialState*             m_initialState = nullptr;
    class Sampler*                  m_sampler = nullptr;
    std::vector<class Particle*>    m_particles = std::vector<class Particle*>();
};
