#include <vector>

class InitialState {
public:
    InitialState(class System* system);
    virtual void setupInitialState() = 0;

protected:
    class System* m_system = nullptr;
    int m_numberOfDimensions = 0;
    int m_numberOfParticles = 0;
    int m_numberOfInputs = 0;
    int m_numberOfHidden = 0;
};
