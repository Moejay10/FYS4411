#include <iostream>
#include <cassert>
#include <random>
#include "Math/random.h"
#include "randomuniform.h"

#include "../NeuralNetworks/network.h"
#include "../system.h"

using namespace std;
using std::cout;
using std::endl;

RandomUniform::RandomUniform(System*    system,
                             int        numberOfDimensions,
                             int        numberOfParticles,
                             int        numberOfHidden) :
    InitialState(system) {
    assert(numberOfDimensions > 0 && numberOfParticles > 0);
    m_numberOfDimensions = numberOfDimensions;
    m_numberOfParticles  = numberOfParticles;
    m_numberOfInputs     = numberOfParticles * numberOfDimensions;
    m_numberOfHidden     = numberOfHidden;

    /* The Initial State class is in charge of everything to do with the
     * initialization of the system; this includes determining the number of
     * particles and the number of dimensions used. To make sure everything
     * works as intended, this information is passed to the system here.
     */

    // Initialize the seed and call the Mersienne algo
    std::random_device rd;
    m_randomEngine = std::mt19937_64(rd());

    m_system->setNumberOfDimensions(numberOfDimensions);
    m_system->setNumberOfParticles(numberOfParticles);
    m_system->setNumberOfHidden(numberOfHidden);
    setupInitialState();
}

void RandomUniform::setupInitialState() {

  // Set up the uniform distribution for x \in [[0, 1]
  std::uniform_real_distribution<double> Uniform(-0.5,0.5);
  std::normal_distribution<double> Normal(0.0,1.0);


  double sigma_initRBM = 0.5; // 0.001;
  std::normal_distribution<double> distribution_initRBM(0, sigma_initRBM);
  //std::uniform_real_distribution<double> distribution_initRBM(0, 0.5);


  vec positions(m_numberOfInputs);
  mat weights(m_numberOfInputs, m_numberOfHidden);
  vec biasA(m_numberOfInputs);
  vec biasB(m_numberOfHidden);

    for (int i = 0; i < m_numberOfInputs; i++) {

      positions(i) = Uniform(m_randomEngine);

      biasA(i)=distribution_initRBM(m_randomEngine);

      for (int j = 0; j < m_numberOfHidden; j++){
          weights(i,j)= distribution_initRBM(m_randomEngine);
      }
    }

    for (int i = 0; i < m_numberOfHidden; i++){
      biasB(i) = distribution_initRBM(m_randomEngine);
    }

    m_system->getNetwork()->setPositions(positions);
    m_system->getNetwork()->setWeights(weights);
    m_system->getNetwork()->setBiasA(biasA);
    m_system->getNetwork()->setBiasB(biasB);


}
