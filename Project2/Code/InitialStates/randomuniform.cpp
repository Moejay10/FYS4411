#include "randomuniform.h"
#include <iostream>
#include <cassert>
#include <random>
#include "Math/random.h"
#include "../particle.h"
#include "../system.h"

using namespace  std;
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
    setupInitialState();
}

void RandomUniform::setupInitialState() {

  // Set up the uniform distribution for x \in [[0, 1]
  std::uniform_real_distribution<double> Uniform(0.0,1.0);
  std::normal_distribution<double> Normal(0.0,1.0);

  std::vector<double> positions = std::vector<double>();

  double sigma_initRBM = 0.001;
  std::normal_distribution<double> distribution_initRBM(0, sigma_initRBM);

  double weights[m_numberOfInputs][m_numberOfHidden];

  std::vector<double> biasA = std::vector<double>();

  std::vector<double> biasB = std::vector<double>();

    for (int i=0; i < m_numberOfInputs; i++) {
      if (m_system->getImportanceSampling()){
            positions.push_back(Normal(m_randomEngine) *sqrt(m_system->getTimeStep()) );
      }
      else{
        positions.push_back(Uniform(m_randomEngine) - 0.5 );
      }

      biasA.push_back(distribution_initRBM(m_randomEngine));

      for (int j=0; j < m_numberOfHidden; j++){
          weights[i][j] = distribution_initRBM(m_randomEngine);
      }

    }

    for (int i = 0; i < numberOfHidden; i++){
      biasB.push_back(distribution_initRBM(m_randomEngine));
    }

    m_NeuralNetwork = new NeuralNetwork();
    m_NeuralNetwork->setPositions(positions);
    m_NeuralNetwork->setWeights(weights);
    m_NeuralNetwork->setBiasA(biasA);
    m_NeuralNetwork->setBiasB(biasB);
}
