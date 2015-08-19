#include <ctime>
#include <string>
#include <iostream>

#include <sys/time.h>

#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"

#include "openmm/System.h"
#include "openmm/Context.h"
#include "openmm/kernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include "LTMD/Analysis.h"
#include "LTMD/Integrator.h"
#include "LTMD/StepKernel.h"

#include <stdio.h>
#include <cstdlib>
#include <string>
#include <iostream>

namespace OpenMM {
	namespace LTMD {
		Integrator::Integrator( double temperature, double frictionCoeff, double stepSize, const Parameters &params )
			: maxEigenvalue( 4.34e5 ), stepsSinceDiagonalize( 0 ), mParameters( params ), mAnalysis( new Analysis ) {
			setTemperature( temperature );
			setFriction( frictionCoeff );
			setStepSize( stepSize );
			setConstraintTolerance( 1e-4 );
			setMinimumLimit( mParameters.minLimit );
			setRandomNumberSeed( ( int ) time( 0 ) );
		}

		Integrator::~Integrator() {
			delete mAnalysis;
		}

		void Integrator::initialize( ContextImpl &contextRef ) {
			context = &contextRef;
			if( context->getSystem().getNumConstraints() > 0 ) {
				throw OpenMMException( "LTMD Integrator does not support constraints" );
			}
			kernel = context->getPlatform().createKernel( StepKernel::Name(), contextRef );
			( ( StepKernel & )( kernel.getImpl() ) ).initialize( contextRef.getSystem(), *this );
		}

		std::vector<std::string> Integrator::getKernelNames() {
			std::vector<std::string> names;
			names.push_back( StepKernel::Name() );
			return names;
		}

		void Integrator::step( int steps ) {
			if( context->getTime() == 0.0 ) {
				computeProjectionVectors();
				mMetropolisPE = context->calcForcesAndEnergy( true, true );
			}

			timeval start, end;
			gettimeofday( &start, 0 );

			mSimpleMinimizations = 0;

			bool isIntegrate = true;
			for( int i = 0; i < steps; i++) {
				if( stepsSinceDiagonalize % mParameters.rediagFreq == 0 ) {
					computeProjectionVectors();
				}

				const double CurrentPE = context->calcForcesAndEnergy( true, true );
				if( isIntegrate ){
					isIntegrate = false;

					IntegrateStep();
					mLastCompleted++;
				}else{
					if( MetropolisTermination(CurrentPE, mMetropolisPE) ) {
						isIntegrate = true;
					}else{
						mSimpleMinimizations++;
						LinearMinimize(CurrentPE);
					}
				}
			}

			// Update Time
			TimeAndCounterStep(mLastCompleted);

			// Print Minimizations
			const unsigned int total = mSimpleMinimizations;

			const double averageSimple = ( double )mSimpleMinimizations / ( double )mLastCompleted;
			const double averageTotal = ( double )total / ( double )mLastCompleted;

			std::cout << "[OpenMM::Minimize] " << total << " total minimizations( "
					  << mSimpleMinimizations << " simple )."
					  << averageTotal << " per-step minimizations( "
					  << averageSimple << " simple ). Steps: "
					  << mLastCompleted << std::endl;

			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Integrator] Total dynamics: " << elapsed << "ms" << std::endl;
		}

		double Integrator::computeKineticEnergy() {
			return ( ( StepKernel & )( kernel.getImpl() ) ).computeKineticEnergy( *context, *this );
		}

		unsigned int Integrator::CompletedSteps() const {
			return mLastCompleted;
		}

		const bool Integrator::MetropolisTermination(const double current, double& original) const {
			if( current < original ) {
				original = current;
				return true;
			}

			const double prob = exp(-( 1.0 / ( BOLTZ * temperature )) * ( current - original ));
			if( SimTKOpenMMUtilities::getUniformlyDistributedRandomNumber() < prob ) {
				original = current;
				return true;
			}

			return false;
		}

		void Integrator::computeProjectionVectors() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			mAnalysis->computeEigenvectorsFull( context->getOwner(), mParameters );
			( ( StepKernel & )( kernel.getImpl() ) ).UpdateEigenvectors( *context, mAnalysis->getEigenvectors() );
			stepsSinceDiagonalize = 0;
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Compute Projection: " << elapsed << "ms" << std::endl;
#endif
		}

		// Kernel Functions
		void Integrator::IntegrateStep() {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			( ( StepKernel & )( kernel.getImpl() ) ).Integrate( *context, *this );
			stepsSinceDiagonalize++;
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Integrate Step: " << elapsed << "ms" << std::endl;
#endif
		}

		void Integrator::TimeAndCounterStep(const unsigned int steps) {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			( ( StepKernel & )( kernel.getImpl() ) ).UpdateTime( *this, steps );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] TimeAndCounter Step: " << elapsed << "ms" << std::endl;
#endif
		}

		void Integrator::LinearMinimize( const double energy ) {
#ifdef PROFILE_INTEGRATOR
			timeval start, end;
			gettimeofday( &start, 0 );
#endif
			( ( StepKernel & )( kernel.getImpl() ) ).LinearMinimize( *context, *this, energy );
#ifdef PROFILE_INTEGRATOR
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[OpenMM::Integrator] Linear Minimize: " << elapsed << "ms" << std::endl;
#endif
		}
	}
}
