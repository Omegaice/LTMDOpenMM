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

			void StepKernel::ProjectionVectors( Integrator &integrator ) {
				//check if projection vectors changed
				bool modesChanged = integrator.getProjVecChanged();

				//projection vectors changed or never allocated
				if( modesChanged || modes == NULL ) {
					int numModes = integrator.getNumProjectionVectors();

					if( numModes == 0 ) {
						throw OpenMMException( "Projection vector size is zero." );
					}

					if( modes != NULL && modes->getSize() != numModes * mParticles ) {
						delete modes;
						delete modeWeights;
						modes = NULL;
						modeWeights = NULL;
					}

					if( modes == NULL ) {
						modes = new CudaArray( *( data.contexts[0] ), numModes * mParticles, sizeof( float4 ), "NormalModes" );
						modeWeights = new CudaArray( *( data.contexts[0] ), ( numModes > data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ? numModes : data.contexts[0]->getNumThreadBlocks() * data.contexts[0]->ThreadBlockSize ), sizeof( float ), "NormalModeWeights" );
						pPosqP = new CudaArray( *( data.contexts[0] ), data.contexts[0]->getPaddedNumAtoms(), sizeof( float4 ), "MidIntegPositions" );
						modesChanged = true;
					}
					if( modesChanged ) {
						int index = 0;
						const std::vector<std::vector<Vec3> > &modeVectors = integrator.getProjectionVectors();
						std::vector<float4> tmp( numModes * mParticles );
						for( int i = 0; i < numModes; i++ ) {
							for( int j = 0; j < mParticles; j++ ) {
								tmp[index++] = make_float4( ( float ) modeVectors[i][j][0], ( float ) modeVectors[i][j][1], ( float ) modeVectors[i][j][2], 0.0f );
							}
						}
						modes->upload( tmp );
						integrator.SetProjectionChanged(false);
					}
				}
			}

			void StepKernel::Integrate( OpenMM::ContextImpl &context, Integrator &integrator ) {
				ProjectionVectors( integrator );

				// Calculate Constants
				const double friction = integrator.getFriction();

				context.updateContextState();
				// Do Step
				kNMLUpdate( &updatemodule,
							data.contexts[0],
							integrator.getStepSize(),
							friction == 0.0f ? 0.0f : 1.0f / friction,
							( float )( BOLTZ * integrator.getTemperature() ),
							integrator.getNumProjectionVectors(), kIterations, *modes, *modeWeights, *NoiseValues );	// TMC setting parameters for this
			}

			void StepKernel::UpdateTime( Integrator &integrator ) {
				data.contexts[0]->setTime( data.contexts[0]->getTime() + integrator.getStepSize() );
				data.contexts[0]->setStepCount( data.contexts[0]->getStepCount() + 1 );
				data.contexts[0]->reorderAtoms();
			}

			void StepKernel::LinearMinimize( OpenMM::ContextImpl &context, Integrator &integrator, const double energy ) {
				ProjectionVectors( integrator );
				kNMLLinearMinimize( &linmodule, data.contexts[0], integrator.getNumProjectionVectors(), integrator.getMaxEigenvalue(), *pPosqP, *modes, *modeWeights );
			}
		}
	}
}
