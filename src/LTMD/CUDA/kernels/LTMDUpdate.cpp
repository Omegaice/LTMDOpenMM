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

void kNMLRejectMinimizationStep( CUmodule *module, CudaContext *cu, CudaArray &oldpos ) {
	int atoms = cu->getNumAtoms();

	CUfunction rejectKernel = cu->getKernel( *module, "kRejectMinimizationStep_kernel" );
	void *rejectArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer() };
	cu->executeKernel( rejectKernel, rejectArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
}

void kNMLAcceptMinimizationStep( CUmodule *module, CudaContext *cu, CudaArray &oldpos ) {
	int atoms = cu->getNumAtoms();

	CUfunction acceptKernel = cu->getKernel( *module, "kAcceptMinimizationStep_kernel" );
	void *acceptArgs[] = {&atoms, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer() };
	cu->executeKernel( acceptKernel, acceptArgs, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize );
}

void kNMLLinearMinimize( CUmodule *module, CudaContext *cu, int numModes, float maxEigenvalue, CudaArray &oldpos, CudaArray &modes, CudaArray &modeWeights ) {
	int atoms = cu->getNumAtoms();
	int paddedatoms = cu->getPaddedNumAtoms();
	float oneoverEig = 1.0f / maxEigenvalue;

	CUfunction linmin1Kernel = cu->getKernel( *module, "kNMLLinearMinimize1_kernel" );
	void *linmin1Args[] = {&atoms, &paddedatoms, &numModes, &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	cu->executeKernel( linmin1Kernel, linmin1Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, cu->ThreadBlockSize * sizeof( float ) );

	CUfunction linmin2Kernel = cu->getKernel( *module, "kNMLLinearMinimize2_kernel" );
	void *linmin2Args[] = {&atoms, &paddedatoms, &numModes, &oneoverEig, &cu->getPosq().getDevicePointer(), &oldpos.getDevicePointer(), &cu->getVelm().getDevicePointer(), &cu->getForce().getDevicePointer(), &modes.getDevicePointer(), &modeWeights.getDevicePointer()};
	cu->executeKernel( linmin2Kernel, linmin2Args, cu->getNumThreadBlocks()*cu->ThreadBlockSize, cu->ThreadBlockSize, numModes * sizeof( float ) );
}
