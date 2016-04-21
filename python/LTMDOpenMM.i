%module ltmdopenmm

%{
#include "LTMD/Integrator.h"

#include "OpenMM.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%import(module="simtk.openmm") "OpenMMSwigHeaders.i"
%include "typemaps.i"

%include "std_string.i"
%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
  %template(vectorf) vector<OpenMM::LTMD::Force>;
}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

%pythonprepend %{
try:
  args=mm.stripUnits(args)
except UnboundLocalError:
  pass
%}

namespace OpenMM {
  namespace LTMD {
    class Integrator : public OpenMM::Integrator {
     public:
      Integrator( const double temperature, const double frictionCoeff, const double stepSize, const OpenMM::LTMD::Parameters &param );
      ~Integrator();
      OpenMM::Integrator& GetBase();
      double getTemperature() const;
      void setTemperature( double temp );
      double getFriction() const;
      void setFriction( double coeff );
      unsigned int getNumProjectionVectors() const;
      double getMinimumLimit() const;
      void setMinimumLimit( double limit );
      bool getProjVecChanged() const;
      const std::vector<std::vector<OpenMM::Vec3> > &getProjectionVectors() const;
      void SetProjectionChanged( bool value );
      void setProjectionVectors( const std::vector<std::vector<OpenMM::Vec3> > &vectors );
      double getMaxEigenvalue() const;
      double computeKineticEnergy();
      int getRandomNumberSeed() const;
      void setRandomNumberSeed( int seed );
      void step( int steps = 1 );
      unsigned int CompletedSteps() const;
      bool minimize( const unsigned int upperbound );
      bool minimize( const unsigned int upperbound, const unsigned int lowerbound );

      double getStepSize() const;
      void setStepSize(double size);
      double getConstraintTolerance() const;
      void setConstraintTolerance(double tol);
    };
  }
}
%include "LTMD/Parameters.h"
