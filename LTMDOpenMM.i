%module ltmdopenmm

%include "std_vector.i"
%include "std_string.i"

%{
#include "LTMD/Integrator.h"
#include "LTMD/Parameters.h"

using namespace OpenMM::LTMD;
%}

namespace std {
  %template(IntVector) vector<int>;
  %template(ForceVector) vector<OpenMM::LTMD::Force>;
}

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
