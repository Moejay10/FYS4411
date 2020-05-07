#include "hamiltonian.h"
#include "../system.h"
#include "../WaveFunctions/wavefunction.h"

Hamiltonian::Hamiltonian(System* system) {
    m_system = system;
}
