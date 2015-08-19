#ifndef OPENMM_LTMD_CUDA_KERNELS_H_
#define OPENMM_LTMD_CUDA_KERNELS_H_

#include "LTMD/StepKernel.h"

#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaPlatform.h"
#include "CudaIntegrationUtilities.h"

static const float KILO                     =    1e3;                      // Thousand
static const float BOLTZMANN                =    1.380658e-23f;            // (J/K)
static const float AVOGADRO                 =    6.0221367e23f;            // ()
static const float RGAS                     =    BOLTZMANN * AVOGADRO;     // (J/(mol K))
static const float BOLTZ                    = ( RGAS / KILO );             // (kJ/(mol K))

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			class StepKernel : public LTMD::StepKernel {
				public:
					StepKernel( std::string name, const OpenMM::Platform &platform, OpenMM::CudaPlatform::PlatformData &data );
					~StepKernel();
					/**
					 * Initialize the kernel, setting up the particle masses.
					 *
					 * @param system     the System this kernel will be applied to
					 * @param integrator the LangevinIntegrator this kernel will be used for
					 */
					void initialize( const OpenMM::System &system, Integrator &integrator );

					void UpdateEigenvectors( OpenMM::ContextImpl &context, const std::vector<std::vector<Vec3> >& vectors );
					void Integrate( OpenMM::ContextImpl &context, Integrator &integrator );
					void UpdateTime( Integrator &integrator, const unsigned int steps );

					void LinearMinimize( OpenMM::ContextImpl &context, Integrator &integrator, const double energy );
					virtual double computeKineticEnergy( OpenMM::ContextImpl &context, Integrator &integrator ) {
						return data.contexts[0]->getIntegrationUtilities().computeKineticEnergy( 0.5 * integrator.getStepSize() );
					}

				private:
					void ProjectionVectors( Integrator &integrator );
				private:
					unsigned int mParticles;
					unsigned int mProjectionVectors;
					OpenMM::CudaPlatform::PlatformData &data;
					CudaArray *modes, *NoiseValues;
					CudaArray *modeWeights;
					CudaArray *MinimizeLambda;
					CudaArray *pPosqP;
					double lastPE;
					int iterations;
					int kIterations;
					CUmodule linmodule;
					CUmodule updatemodule;
			};
		}
	}
}

#endif // OPENMM_LTMD_CUDA_KERNELS_H_
