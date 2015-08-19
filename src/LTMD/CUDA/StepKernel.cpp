#include "LTMD/CUDA/StepKernel.h"

#include <cmath>
#include "SimTKOpenMMUtilities.h"
#include "OpenMM.h"
#include "CudaIntegrationUtilities.h"
#include "CudaKernels.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include "LTMD/CUDA/KernelSources.h"
#include "LTMD/Integrator.h"
#include <stdlib.h>
#include <iostream>
using namespace std;

using namespace OpenMM;

extern void kGenerateRandoms ( CudaContext *gpu );
void kNMLUpdate ( CUmodule *module, CudaContext *gpu, float deltaT, float tau, float kT, int numModes, int &iterations, CudaArray &modes, CudaArray &modeWeights, CudaArray &noiseValues );
void kNMLLinearMinimize ( CUmodule *module, CudaContext *gpu, int numModes, float maxEigenvalue, CudaArray &oldpos, CudaArray &modes, CudaArray &modeWeights );

double drand() {/* uniform distribution, (0..1] */
	return ( rand() + 1.0 ) / ( RAND_MAX + 1.0 );
}
double random_normal() {/* normal distribution, centered on 0, std dev 1 */
	return sqrt( -2 * log( drand() ) ) * cos( 2 * M_PI * drand() );
}

namespace OpenMM {
	namespace LTMD {
		namespace CUDA {
			StepKernel::StepKernel( std::string name, const Platform &platform, CudaPlatform::PlatformData &data ) : LTMD::StepKernel( name, platform ),
				data( data ), modes( NULL ), modeWeights( NULL ) {
				iterations = 0;
				kIterations = 0;
			}

			StepKernel::~StepKernel() {
				if( modes != NULL ) {
					delete modes;
				}
				if( modeWeights != NULL ) {
					delete modeWeights;
				}
			}

			void StepKernel::initialize( const System &system, Integrator &integrator ) {
				data.contexts[0]->initialize();
				linmodule = data.contexts[0]->createModule( KernelSources::linearMinimizers );
				updatemodule = data.contexts[0]->createModule( KernelSources::NMLupdates );

				mParticles = data.contexts[0]->getNumAtoms();

				NoiseValues = new CudaArray( *( data.contexts[0] ), mParticles, sizeof( float4 ), "NoiseValues" );
				std::vector<float4> tmp( mParticles );
				for( size_t i = 0; i < mParticles; i++ ) {
					tmp[i] = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
				}
				NoiseValues->upload( tmp );

				data.contexts[0]->getIntegrationUtilities().initRandomNumberGenerator( integrator.getRandomNumberSeed() );
			}

			void StepKernel::UpdateEigenvectors(OpenMM::ContextImpl &context, const std::vector<std::vector<Vec3> >& vectors) {
				mProjectionVectors = vectors.size();

				if( modes != NULL && modes->getSize() != mProjectionVectors * mParticles ) {
					delete modes;
					delete modeWeights;
					modes = NULL;
					modeWeights = NULL;
				}

				if( modes == NULL ) {
					modes = new CudaArray( *( data.contexts[0] ), mProjectionVectors * mParticles, sizeof( float4 ), "NormalModes" );
					modeWeights = new CudaArray( *( data.contexts[0] ), ( mProjectionVectors > data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ? mProjectionVectors : data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ), sizeof( float ), "NormalModeWeights" );
					pPosqP = new CudaArray( *( data.contexts[0] ), data.contexts[0]->getPaddedNumAtoms(), sizeof( float4 ), "MidIntegPositions" );
				}

				int index = 0;
				std::vector<float4> tmp( mProjectionVectors * mParticles );
				for( int i = 0; i < mProjectionVectors; i++ ) {
					for( int j = 0; j < mParticles; j++ ) {
						tmp[index++] = make_float4( ( float ) vectors[i][j][0], ( float ) vectors[i][j][1], ( float ) vectors[i][j][2], 0.0f );
					}
				}
				modes->upload( tmp );
			}

			void StepKernel::Integrate( OpenMM::ContextImpl &context, Integrator &integrator ) {
				// Calculate Constants
				const double friction = integrator.getFriction();

				context.updateContextState();
				// Do Step
				kNMLUpdate( &updatemodule,
							data.contexts[0],
							integrator.getStepSize(),
							friction == 0.0f ? 0.0f : 1.0f / friction,
							( float )( BOLTZ * integrator.getTemperature() ),
							mProjectionVectors, kIterations, *modes, *modeWeights, *NoiseValues );	// TMC setting parameters for this
			}

			void StepKernel::UpdateTime( Integrator &integrator, const unsigned int steps ) {
				data.contexts[0]->setTime( data.contexts[0]->getTime() + integrator.getStepSize() * steps );
				data.contexts[0]->setStepCount( data.contexts[0]->getStepCount() + steps );
			}

			void StepKernel::LinearMinimize( OpenMM::ContextImpl &context, Integrator &integrator, const double energy ) {
				kNMLLinearMinimize( &linmodule, data.contexts[0], mProjectionVectors, integrator.getMaxEigenvalue(), *pPosqP, *modes, *modeWeights );
			}
		}
	}
}
