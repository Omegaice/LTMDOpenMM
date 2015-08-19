#ifndef OPENMM_LTMD_STEPKERNEL_H_
#define OPENMM_LTMD_STEPKERNEL_H_

#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include "LTMD/Integrator.h"
#include "openmm/internal/ContextImpl.h"

namespace OpenMM {
	namespace LTMD {
		/**
		 * This kernel is invoked by NMLIntegrator to take one time step.
		 */
		class StepKernel : public OpenMM::KernelImpl {
			public:
				static std::string Name() {
					return "IntegrateNMLStep";
				}
				StepKernel( std::string name, const OpenMM::Platform &platform ) : KernelImpl( name, platform ) { }

				/**
				 * Initialize the kernel.
				 *
				 * @param system     the System this kernel will be applied to
				 * @param integrator the NMLIntegrator this kernel will be used for
				 */
				virtual void initialize( const OpenMM::System &system, Integrator &integrator ) = 0;

				virtual void Integrate( OpenMM::ContextImpl &context, Integrator &integrator ) = 0;
				virtual void UpdateTime( Integrator &integrator, const unsigned int steps ) = 0;

				virtual double computeKineticEnergy( OpenMM::ContextImpl &context, Integrator &integrator ) = 0;

				virtual void LinearMinimize( OpenMM::ContextImpl &context, Integrator &integrator, const double energy ) = 0;
		};
	}
}

#endif // OPENMM_LTMD_STEPKERNEL_H_
