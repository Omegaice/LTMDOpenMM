#include "CudaIntegrationUtilities.h"
#include "CudaContext.h"
#include "CudaArray.h"
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <vector_functions.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <stdlib.h>
using namespace std;
using namespace OpenMM;


// CPU code
void kNMLUpdate( CUmodule *module, CudaContext *cu, float deltaT, float tau, float kT, int numModes, int &iterations, CudaArray &modes, CudaArray &modeWeights, CudaArray &noiseVal ) {
	int atoms = cu->getNumAtoms();
	int paddednumatoms = cu->getPaddedNumAtoms();
	int randomIndex = cu->getIntegrationUtilities().prepareRandomNumbers(cu->getPaddedNumAtoms());

	CUfunction update1Kernel = cu->getKernel( *module, "kNMLUpdate1_kernel" );
	void *update1Args[] = {
		&atoms, &paddednumatoms, &tau, &deltaT, &kT,
		&cu->getPosq().getDevicePointer(), &noiseVal.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex
	};
	cu->executeKernel( update1Kernel, update1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );

	CUfunction update2Kernel = cu->getKernel( *module, "kNMLUpdate2_kernel" );
	void *update2Args[] = {&atoms, &numModes, &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	cu->executeKernel( update2Kernel, update2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );

	CUfunction update3Kernel = cu->getKernel( *module, "kNMLUpdate3_kernel" );
	void *update3Args[] = {&atoms, &numModes, &deltaT, &cu->getPosq().getDevicePointer(), &cu->getVelm().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer(), &noiseVal.getDevicePointer()};
	cu->executeKernel( update3Kernel, update3Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes * sizeof( float ) );

}

void kNMLLinearMinimize( CUmodule *module, CudaContext *cu, int numModes, float maxEigenvalue, CudaArray &oldpos, CudaArray &modes, CudaArray &modeWeights, float kT, float eCurrent, CudaArray &passed ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	int randomIndex = cu->getIntegrationUtilities().prepareRandomNumbers(1);
	float oneoverEig = 1.0f / maxEigenvalue;
	int blockSize = cu->ThreadBlockSize;
    int gridSize = std::min((cu->getNumThreadBlocks()*cu->ThreadBlockSize+blockSize-1)/blockSize, cu->getNumThreadBlocks());
	int eCount = cu->getEnergyBuffer().getSize();

	CUfunction linmin1Kernel = cu->getKernel( *module, "kNMLMinimize" );
	void *linmin1Args[] = {
		&blockSize, &gridSize, &kT, &eCurrent, &passed.getDevicePointer(), &cu->getIntegrationUtilities().getRandom().getDevicePointer(), &randomIndex, &cu->getEnergyBuffer().getDevicePointer(), &eCount, &atoms, &paddedatoms, &numModes, &oneoverEig,
		&cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()
	};
	cu->executeKernel( linmin1Kernel, linmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );
}
