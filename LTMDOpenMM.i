%module ltmdopenmm

%include "std_vector.i"

%{
#include "LTMD/Integrator.h"
#include "LTMD/Parameters.h"

using namespace OpenMM::LTMD;
%}
%include "LTMD/Integrator.h"
%include "LTMD/Parameters.h"
