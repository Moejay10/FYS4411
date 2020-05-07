#include "initialstate.h"
#include <random>
#include <iostream>

class RandomUniform : public InitialState {
public:
    RandomUniform(System* system, int numberOfDimensions, int numberOfParticles, int numberOfHidden, double sigma);
    void setupInitialState();

private:
  std::mt19937_64 m_randomEngine; // For the distributions
};
