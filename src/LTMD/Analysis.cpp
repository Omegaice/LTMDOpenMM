/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include "openmm/Vec3.h"
#include <sys/time.h>
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>

#include "OpenMM.h"
#include "LTMD/Math.h"
#include "LTMD/Analysis.h"
#include "LTMD/Integrator.h"

#include "tnt_array2d_utils.h"

namespace OpenMM {
	namespace LTMD {
		const unsigned int ConservedDegreesOfFreedom = 6;

		// Function Declarations
	  void WriteS(const TNT::Array2D<double>& S);
		void WriteHessian( const TNT::Array2D<double>& H );
		void WriteBlockEigs( const TNT::Array2D<double>& H );
		void WriteModes( const TNT::Array2D<double>& U, const unsigned int modes );

		// Function Implementations
		unsigned int Analysis::blockNumber( int p ) {
			unsigned int block = 0;
			while( block != blocks.size() && blocks[block] <= p ) {
				block++;
			}
			return block - 1;
		}

		bool Analysis::inSameBlock( int p1, int p2, int p3 = -1, int p4 = -1 ) {
			if( blockNumber( p1 ) != blockNumber( p2 ) ) {
				return false;
			}

			if( p3 != -1 && blockNumber( p3 ) != blockNumber( p1 ) ) {
				return false;
			}

			if( p4 != -1 && blockNumber( p4 ) != blockNumber( p1 ) ) {
				return false;
			}

			return true;   // They're all the same!
		}

		void Analysis::computeEigenvectorsFull( ContextImpl &contextImpl, const Parameters& params ) {
			timeval start, end;
			gettimeofday( &start, 0 );

			timeval tp_begin, tp_hess, tp_diag, tp_e, tp_s, tp_q, tp_u;

			gettimeofday( &tp_begin, NULL );
			Context &context = contextImpl.getOwner();
			State state = context.getState( State::Positions | State::Forces );
			vector<Vec3> positions = state.getPositions();
			
			/*********************************************************************/
			/*                                                                   */
			/* Block Hessian Code (Cickovski/Sweet)                              */
			/*                                                                   */
			/*********************************************************************/

			// Initial residue data (where in OpenMM?)

			// For now, since OpenMM input files do not contain residue information
			// I am assuming that they will always start with the N-terminus, just for testing.
			// This is true for the villin.xml but may not be true in the future.
			// need it to parallelize.

			if( !mInitialized ) {
				Initialize( context, params );
			}
			
			int n = 3 * mParticleCount;

			// Copy the positions.
			bool isBlockDoublePrecision = blockContext->getPlatform().supportsDoublePrecision();
			vector<Vec3> blockPositions;
			for( unsigned int i = 0; i < mParticleCount; i++ ) {
				Vec3 atom( state.getPositions()[i][0], state.getPositions()[i][1], state.getPositions()[i][2] );
				blockPositions.push_back( atom );
			}

			blockContext->setPositions( blockPositions );
			/*********************************************************************/

			TNT::Array2D<double> h( n, n, 0.0 );
			vector<Vec3> initialBlockPositions( blockPositions );
			for( unsigned int i = 0; i < mLargestBlockSize; i++ ) {
				// Perturb the ith degree of freedom in EACH block
				// Note: not all blocks will have i degrees, we have to check for this
				for( unsigned int j = 0; j < blocks.size(); j++ ) {
					unsigned int dof_to_perturb = 3 * blocks[j] + i;
					unsigned int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3] - params.blockDelta;
				}

				blockContext->setPositions( blockPositions );
				vector<Vec3> forces1 = blockContext->getState( State::Forces ).getForces();

				// Now, do it again...
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3] + params.blockDelta;
				}

				blockContext->setPositions( blockPositions );
				vector<Vec3> forces2 = blockContext->getState( State::Forces ).getForces();

				// revert block positions
				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					blockPositions[atom_to_perturb][dof_to_perturb % 3] = initialBlockPositions[atom_to_perturb][dof_to_perturb % 3];

				}

				for( int j = 0; j < blocks.size(); j++ ) {
					int dof_to_perturb = 3 * blocks[j] + i;
					int atom_to_perturb = dof_to_perturb / 3;  // integer trunc

					// Cases to not perturb, in this case just skip the block
					if( j == blocks.size() - 1 && atom_to_perturb >= mParticleCount ) {
						continue;
					}
					if( j != blocks.size() - 1 && atom_to_perturb >= blocks[j + 1] ) {
						continue;
					}

					int col = dof_to_perturb; //(atom_to_perturb*3)+(dof_to_perturb % 3);
					int row = 0;

					int start_dof = 3 * blocks[j];
					int end_dof;
					if( j == blocks.size() - 1 ) {
						end_dof = 3 * mParticleCount;
					} else {
						end_dof = 3 * blocks[j + 1];
					}

					for( int k = start_dof; k < end_dof; k++ ) {
						double blockscale = 1.0 / ( 2 * params.blockDelta * sqrt( mParticleMass[atom_to_perturb] * mParticleMass[k / 3] ) );
						h[k][col] = ( forces1[k / 3][k % 3] - forces2[k / 3][k % 3] ) * blockscale;
					}
				}
			}

			gettimeofday( &tp_hess, NULL );
			cout << "Time to compute hessian: " << ( tp_hess.tv_sec - tp_begin.tv_sec ) << endl;

			// Make sure it is exactly symmetric.
			for( int i = 0; i < n; i++ ) {
				for( int j = 0; j < i; j++ ) {
					double avg = 0.5f * ( h[i][j] + h[j][i] );
					h[i][j] = avg;
				}
			}

			WriteHessian( h );

			// Diagonalize each block Hessian, get Eigenvectors
			// Note: The eigenvalues will be placed in one large array, because
			//       we must sort them to get k

			TNT::Array1D<double> block_eigval( n, 0.0 );
			TNT::Array2D<double> block_eigvec( n, n, 0.0 );
			
			#pragma omp parallel for
			for( int i = 0; i < blocks.size(); i++ ) {
				DiagonalizeBlock( i, h, positions, block_eigval, block_eigvec );
			}

			gettimeofday( &tp_diag, NULL );
			cout << "Time to diagonalize block hessian: " << ( tp_diag.tv_sec - tp_hess.tv_sec ) << endl;
			//***********************************************************
			// This section here is only to find the cuttoff eigenvalue.
			// First sort the eigenvectors by the absolute value of the eigenvalue.

			// sort all eigenvalues by absolute magnitude to determine cutoff
			vector<pair<double, int> > sortedEvalues( n );
			for( int i = 0; i < n; i++ ) {
				sortedEvalues[i] = make_pair( fabs( block_eigval[i] ), i );
			}
			sort( sortedEvalues.begin(), sortedEvalues.end() );

			int max_eigs = params.bdof * blocks.size();
			double cutEigen = sortedEvalues[max_eigs].first;  // This is the cutoff eigenvalue
			
			// get cols of all eigenvalues under cutoff
			vector<int> selectedEigsCols;
			for( int i = 0; i < n; i++ ) {
				if( fabs( block_eigval[i] ) < cutEigen ) {
					selectedEigsCols.push_back( i );
				}
			}
			
			// we may select fewer eigs if there are duplicate eigenvalues
			const int m = selectedEigsCols.size();

			// Inefficient, needs to improve.
			// Basically, just setting up E and E^T by
			// copying values from bigE.
			// Again, right now I'm only worried about
			// correctness plus this time will be marginal compared to
			// diagonalization.
			TNT::Array2D<double> E( n, m, 0.0 );
			TNT::Array2D<double> E_transpose( m, n, 0.0 );
			for( int i = 0; i < m; i++ ) {
				int eig_col = selectedEigsCols[i];
				for( int j = 0; j < n; j++ ) {
					E_transpose[i][j] = block_eigvec[j][eig_col];
					E[j][i] = block_eigvec[j][eig_col];
				}
			}

			gettimeofday( &tp_e, NULL );
			cout << "Time to compute E: " << ( tp_e.tv_sec - tp_diag.tv_sec ) << endl;

			WriteBlockEigs( E );

			//*****************************************************************
			// Compute S, which is equal to E^T * H * E.
			TNT::Array2D<double> S( m, m, 0.0 );
			TNT::Array2D<double> HE(n , m, 0.0);
			// Compute eps.
			const double eps = params.sDelta;

			// Make a temp copy of positions.
			vector<Vec3> tmppos( positions );

			// Loop over i.
			for( unsigned int k = 0; k < m; k++ ) {
				// Perturb positions.
				int pos = 0;

				// forward perturbations
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] + eps * E[3 * i + j][k] / sqrt( mParticleMass[i] );
						pos++;
					}
				}
				context.setPositions( tmppos );

				// Calculate F(xi).
				vector<Vec3> forces_forward = context.getState( State::Forces ).getForces();

				// backward perturbations
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j] - eps * E[3 * i + j][k] / sqrt( mParticleMass[i] );
					}
				}
				context.setPositions( tmppos );

				// Calculate forces
				vector<Vec3> forces_backward = context.getState( State::Forces ).getForces();

				for( int i = 0; i < n; i++ ) {
					const double scaleFactor = sqrt( mParticleMass[i / 3] ) * 2.0 * eps;
					HE[i][k] = ( forces_forward[i / 3][i % 3] - forces_backward[i / 3][i % 3] ) / scaleFactor;
				}

				// restore positions
				for( unsigned int i = 0; i < mParticleCount; i++ ) {
					for( unsigned int j = 0; j < 3; j++ ) {
						tmppos[i][j] = positions[i][j];
					}
				}
			}

			// *****************************************************************
			// restore unperturbed positions
			context.setPositions( positions );

			MatrixMultiply( E_transpose, HE, S );


			WriteS(S);

			// make S symmetric
			for( unsigned int i = 0; i < S.dim1(); i++ ) {
				for( unsigned int j = 0; j < S.dim2(); j++ ) {
					double avg = 0.5f * ( S[i][j] + S[j][i] );
					S[i][j] = avg;
					S[j][i] = avg;
				}
			}

			gettimeofday( &tp_s, NULL );
			cout << "Time to compute S: " << ( tp_s.tv_sec - tp_e.tv_sec ) << endl;

			// Diagonalizing S by finding eigenvalues and eigenvectors...
			TNT::Array1D<double> dS( m, 0.0 );
			TNT::Array2D<double> q( m, m, 0.0 );
			FindEigenvalues( S, dS, q );

			// Sort by ABSOLUTE VALUE of eigenvalues.
			sortedEvalues.clear();
			sortedEvalues.resize( dS.dim() );
			for( int i = 0; i < dS.dim(); i++ ) {
				sortedEvalues[i] = make_pair( fabs( dS[i] ), i );
			}
			sort( sortedEvalues.begin(), sortedEvalues.end() );

			TNT::Array2D<double> Q( q.dim2(), q.dim1(), 0.0 );
			for( int i = 0; i < sortedEvalues.size(); i++ )
				for( int j = 0; j < q.dim2(); j++ ) {
					Q[j][i] = q[j][sortedEvalues[i].second];
				}
			maxEigenvalue = sortedEvalues[dS.dim() - 1].first;

			gettimeofday( &tp_q, NULL );
			cout << "Time to compute Q: " << ( tp_q.tv_sec - tp_s.tv_sec ) << endl;

			// Compute U, set of approximate eigenvectors.
			// U = E*Q.
			TNT::Array2D<double> U( E.dim1(), Q.dim2(), 0.0 );
			MatrixMultiply( E, Q, U );
			
			gettimeofday( &tp_u, NULL );
			cout << "Time to compute U: " << ( tp_u.tv_sec - tp_q.tv_sec ) << endl;

			const unsigned int modes = params.modes;

			WriteModes( U, modes );

			eigenvectors.resize( modes, vector<Vec3>( mParticleCount ) );
			for( unsigned int i = 0; i < modes; i++ ) {
				for( unsigned int j = 0; j < mParticleCount; j++ ) {
					eigenvectors[i][j] = Vec3( U[3 * j][i], U[3 * j + 1][i], U[3 * j + 2][i] );
				}
			}

			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Analysis] Compute Eigenvectors: " << elapsed << "ms" << std::endl;
		}

	  void WriteS(const TNT::Array2D<double>& S)
	  {
#ifdef VALIDATION
	    std::ofstream file("s.txt");
	    file.precision(10);
	    for( unsigned int i = 0; i < S.dim2(); i++ ) {
	      for( unsigned int j = 0; j < S.dim1(); j++ ) {
		file << j << " " << i << " " << S[j][i] << std::endl;
	      }
	    }
#endif
	  }
		
		void WriteHessian( const TNT::Array2D<double>& H ) {
#ifdef VALIDATION
			std::ofstream file( "block_hessian.txt" );
			file.precision( 10 );
			for( unsigned int i = 0; i < H.dim2(); i++ ) {
				for( unsigned int j = 0; j < H.dim1(); j++ ) {
					file << j << " " << i << " " << H[j][i] << std::endl;
				}
			}
#endif
		}
		
		
		void WriteBlockEigs( const TNT::Array2D<double>& matrix ) {
#ifdef VALIDATION
			std::ofstream file( "block_eigs.txt" );
			file.precision( 10 );
			for( unsigned int i = 0; i < matrix.dim2(); i++ ) {
				for( unsigned int j = 0; j < matrix.dim1(); j++ ) {
					file << j << " " << i << " " << matrix[j][i] << std::endl;
				}
			}
#endif
		}

		void WriteModes( const TNT::Array2D<double>& U, const unsigned int modes ) {
#ifdef VALIDATION
			std::ofstream file( "eigenvectors.txt" );
			file.precision( 10 );
			for( unsigned int i = 0; i < modes; i++ ) {
				for( unsigned int j = 0; j < U.dim1(); j++ ) {
					file << j << " " << i << " " << U[j][i] << std::endl;
				}
			}
#endif
		}

		void Analysis::Initialize( Context &context, const Parameters &params ) {
#ifdef PROFILE_ANALYSIS
			timeval start, end;
			gettimeofday( &start, 0 );
#endif

			// Get Current System
			System &system = context.getSystem();
			
			// Store Particle Information
			mParticleCount = context.getState( State::Positions ).getPositions().size();
			
			mParticleMass.reserve( mParticleCount );
			for( unsigned int i = 0; i < mParticleCount; i++ ){
				mParticleMass.push_back( system.getParticleMass( i ) );
			}	
			
			// Create New System
			System *blockSystem = new System();
			cout << "res per block " << params.res_per_block << endl;
			for( int i = 0; i < mParticleCount; i++ ) {
				blockSystem->addParticle( mParticleMass[i] );
			}

			int block_start = 0;
			for( int i = 0; i < params.residue_sizes.size(); i++ ) {
				if( i % params.res_per_block == 0 ) {
					blocks.push_back( block_start );
				}
				block_start += params.residue_sizes[i];
			}

			for( int i = 1; i < blocks.size(); i++ ) {
				int block_size = blocks[i] - blocks[i - 1];
				if( block_size > mLargestBlockSize ) {
					mLargestBlockSize = block_size;
				}
			}

			mLargestBlockSize *= 3; // degrees of freedom in the largest block
			cout << "blocks " << blocks.size() << endl;
			cout << blocks[blocks.size() - 1] << endl;

			// Creating a whole new system called the blockSystem.
			// This system will only contain bonds, angles, dihedrals, and impropers
			// between atoms in the same block.
			// Also contains pairwise force terms which are zeroed out for atoms
			// in different blocks.
			// This necessitates some copying from the original system, but is required
			// because OpenMM populates all data when it reads XML.
			// Copy all atoms into the block system.

			// Copy the center of mass force.
			cout << "adding forces..." << endl;
			for( int i = 0; i < params.forces.size(); i++ ) {
				string forcename = params.forces[i].name;
				cout << "Adding force " << forcename << " at index " << params.forces[i].index << endl;
				if( forcename == "CenterOfMass" ) {
					blockSystem->addForce( &system.getForce( params.forces[i].index ) );
				} else if( forcename == "Bond" ) {
					// Create a new harmonic bond force.
					// This only contains pairs of atoms which are in the same block.
					// I have to iterate through each bond from the old force, then
					// selectively add them to the new force based on this condition.
					HarmonicBondForce *hf = new HarmonicBondForce();
					const HarmonicBondForce *ohf = dynamic_cast<const HarmonicBondForce *>( &system.getForce( params.forces[i].index ) );
					for( int i = 0; i < ohf->getNumBonds(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2;
						double length, k;
						ohf->getBondParameters( i, particle1, particle2, length, k );
						if( inSameBlock( particle1, particle2 ) ) {
							hf->addBond( particle1, particle2, length, k );
						}
					}
					blockSystem->addForce( hf );
				} else if( forcename == "Angle" ) {
					// Same thing with the angle force....
					HarmonicAngleForce *af = new HarmonicAngleForce();
					const HarmonicAngleForce *ahf = dynamic_cast<const HarmonicAngleForce *>( &system.getForce( params.forces[i].index ) );
					for( int i = 0; i < ahf->getNumAngles(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2, particle3;
						double angle, k;
						ahf->getAngleParameters( i, particle1, particle2, particle3, angle, k );
						if( inSameBlock( particle1, particle2, particle3 ) ) {
							af->addAngle( particle1, particle2, particle3, angle, k );
						}
					}
					blockSystem->addForce( af );
				} else if( forcename == "Dihedral" ) {
					// And the dihedrals....
					PeriodicTorsionForce *ptf = new PeriodicTorsionForce();
					const PeriodicTorsionForce *optf = dynamic_cast<const PeriodicTorsionForce *>( &system.getForce( params.forces[i].index ) );
					for( int i = 0; i < optf->getNumTorsions(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2, particle3, particle4, periodicity;
						double phase, k;
						optf->getTorsionParameters( i, particle1, particle2, particle3, particle4, periodicity, phase, k );
						if( inSameBlock( particle1, particle2, particle3, particle4 ) ) {
							ptf->addTorsion( particle1, particle2, particle3, particle4, periodicity, phase, k );
						}
					}
					blockSystem->addForce( ptf );
				} else if( forcename == "Improper" ) {
					// And the impropers....
					RBTorsionForce *rbtf = new RBTorsionForce();
					const RBTorsionForce *orbtf = dynamic_cast<const RBTorsionForce *>( &system.getForce( params.forces[i].index ) );
					for( int i = 0; i < orbtf->getNumTorsions(); i++ ) {
						// For our system, add bonds between atoms in the same block
						int particle1, particle2, particle3, particle4;
						double c0, c1, c2, c3, c4, c5;
						orbtf->getTorsionParameters( i, particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5 );
						if( inSameBlock( particle1, particle2, particle3, particle4 ) ) {
							rbtf->addTorsion( particle1, particle2, particle3, particle4, c0, c1, c2, c3, c4, c5 );
						}
					}
					blockSystem->addForce( rbtf );
				} else if( forcename == "Nonbonded" ) {
					// This is a custom nonbonded pairwise force and
					// includes terms for both LJ and Coulomb.
					// Note that the step term will go to zero if block1 does not equal block 2,
					// and will be one otherwise.
					CustomBondForce *cbf = new CustomBondForce( "4*eps*((sigma/r)^12-(sigma/r)^6)+138.935456*q/r" );
					const NonbondedForce *nbf = dynamic_cast<const NonbondedForce *>( &system.getForce( params.forces[i].index ) );

					cbf->addPerBondParameter( "q" );
					cbf->addPerBondParameter( "sigma" );
					cbf->addPerBondParameter( "eps" );

					// store exceptions
					// exceptions[p1][p2] = params
					map<int, map<int, vector<double> > > exceptions;

					for( int i = 0; i < nbf->getNumExceptions(); i++ ) {
						int p1, p2;
						double q, sig, eps;
						nbf->getExceptionParameters( i, p1, p2, q, sig, eps );
						if( inSameBlock( p1, p2 ) ) {
							vector<double> params;
							params.push_back( q );
							params.push_back( sig );
							params.push_back( eps );
							if( exceptions.count( p1 ) == 0 ) {
								map<int, vector<double> > pair_exception;
								pair_exception[p2] = params;
								exceptions[p1] = pair_exception;
							} else {
								exceptions[p1][p2] = params;
							}
						}
					}

					// add particle params
					// TODO: iterate over block dimensions to reduce to O(b^2 N_b)
					for( int i = 0; i < nbf->getNumParticles() - 1; i++ ) {
						for( int j = i + 1; j < nbf->getNumParticles(); j++ ) {
							if( !inSameBlock( i, j ) ) {
								continue;
							}
							// we have an exception -- 1-4 modified interactions, etc.
							if( exceptions.count( i ) == 1 && exceptions[i].count( j ) == 1 ) {
								vector<double> params = exceptions[i][j];
								cbf->addBond( i, j, params );
							}
							// no exception, normal interaction
							else {
								vector<double> params;
								double q1, q2, eps1, eps2, sigma1, sigma2, q, eps, sigma;

								nbf->getParticleParameters( i, q1, sigma1, eps1 );
								nbf->getParticleParameters( j, q2, sigma2, eps2 );

								q = q1 * q2;
								sigma = 0.5 * ( sigma1 + sigma2 );
								eps = sqrt( eps1 * eps2 );

								params.push_back( q );
								params.push_back( sigma );
								params.push_back( eps );

								cbf->addBond( i, j, params );
							}
						}
					}

					blockSystem->addForce( cbf );
				} else {
					cout << "Unknown Force: " << forcename << endl;
				}
			}
			cout << "done." << endl;

			VerletIntegrator *integ = new VerletIntegrator( 0.000001 );
			if( blockContext ) {
				delete blockContext;
			}
			
			switch( params.BlockDiagonalizePlatform ){
				case Preference::Reference:{
					blockContext = new Context( *blockSystem, *integ, Platform::getPlatformByName( "Reference" ) );
					break;
				}
				case Preference::OpenCL:{
					blockContext = new Context( *blockSystem, *integ, Platform::getPlatformByName( "OpenCL" ) );
					break;
				}
				case Preference::CUDA:{
					blockContext = new Context( *blockSystem, *integ, Platform::getPlatformByName( "Cuda" ) );
					break;
				}
			}
			

			mInitialized = true;

#ifdef PROFILE_ANALYSIS
			gettimeofday( &end, 0 );
			double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
			std::cout << "[Analysis] Initialize: " << elapsed << "ms" << std::endl;
#endif
		}
		
		void Analysis::DiagonalizeBlock( const unsigned int block, const TNT::Array2D<double>& hessian, 
			const std::vector<Vec3>& positions, TNT::Array1D<double>& eval, TNT::Array2D<double>& evec ) {
					
			printf( "Diagonalizing Block: %d\n", block );
			
			// 1. Determine the starting and ending index for the block
			//    This means that the upper left corner of the block will be at (startatom, startatom)
			//    And the lower right corner will be at (endatom, endatom)
			const int startatom = 3 * blocks[block];
			int endatom = 3 * mParticleCount - 1 ;
			
			if( block != ( blocks.size() - 1 ) ) {
				endatom = 3 * blocks[block + 1] - 1;
			}

			const int size = endatom - startatom + 1;

			// 2. Get the block Hessian Hii
			//    Right now I'm just doing a copy from the big Hessian
			//    There's probably a more efficient way but for now I just want things to work..
			TNT::Array2D<double> h_tilde( size, size, 0.0 );
			for( int j = startatom; j <= endatom; j++ ) {
				for( int k = startatom; k <= endatom; k++ ) {
					h_tilde[k - startatom][j - startatom] = hessian[k][j];
				}
			}

			// 3. Diagonalize the block Hessian only, and get eigenvectors
			TNT::Array1D<double> di( size, 0.0 );
			TNT::Array2D<double> Qi( size, size, 0.0 );
			FindEigenvalues( h_tilde, di, Qi );

			// sort eigenvalues by absolute magnitude
			vector<pair<double, int> > sortedEvalPairs( size );
			for( int j = 0; j < size; j++ ) {
				sortedEvalPairs.at( j ) = make_pair( fabs( di[j] ), j );
			}
			sort( sortedEvalPairs.begin(), sortedEvalPairs.end() );

			// find geometric dof
			TNT::Array2D<double> Qi_gdof( size, size, 0.0 );

			Vec3 pos_center( 0.0, 0.0, 0.0 );
			double totalmass = 0.0;

			for( int j = startatom; j <= endatom; j += 3 ) {
				double mass = mParticleMass[ j / 3 ];
				pos_center += positions[j / 3] * mass;
				totalmass += mass;
			}

			double norm = sqrt( totalmass );

			// actual center
			pos_center *= 1.0 / totalmass;

			// create geometric dof vectors
			// iterating over rows and filling in values for 6 vectors as we go
			for( int j = 0; j < size; j += 3 ) {
				double atom_index = ( startatom + j ) / 3;
				double mass = mParticleMass[atom_index];
				double factor = sqrt( mass ) / norm;

				// translational
				Qi_gdof[j][0]   = factor;
				Qi_gdof[j + 1][1] = factor;
				Qi_gdof[j + 2][2] = factor;

				// rotational
				// cross product of rotation axis and vector to center of molecule
				// x-axis (b1=1) ja3-ka2
				// y-axis (b2=1) ka1-ia3
				// z-axis (b3=1) ia2-ja1
				Vec3 diff = positions[atom_index] - pos_center;
				// x
				Qi_gdof[j + 1][3] =  diff[2] * factor;
				Qi_gdof[j + 2][3] = -diff[1] * factor;

				// y
				Qi_gdof[j][4]   = -diff[2] * factor;
				Qi_gdof[j + 2][4] =  diff[0] * factor;

				// z
				Qi_gdof[j][5]   =  diff[1] * factor;
				Qi_gdof[j + 1][5] = -diff[0] * factor;
			}

			// normalize first rotational vector
			double rotnorm = 0.0;
			for( int j = 0; j < size; j++ ) {
				rotnorm += Qi_gdof[j][3] * Qi_gdof[j][3];
			}

			rotnorm = 1.0 / sqrt( rotnorm );

			for( int j = 0; j < size; j++ ) {
				Qi_gdof[j][3] = Qi_gdof[j][3] * rotnorm;
			}

			// orthogonalize rotational vectors 2 and 3
			for( int j = 4; j < ConservedDegreesOfFreedom; j++ ) { // <-- vector we're orthogonalizing
				for( int k = 3; k < j; k++ ) { // <-- vectors we're orthognalizing against
					double dot_prod = 0.0;
					for( int l = 0; l < size; l++ ) {
						dot_prod += Qi_gdof[l][k] * Qi_gdof[l][j];
					}
					for( int l = 0; l < size; l++ ) {
						Qi_gdof[l][j] = Qi_gdof[l][j] - Qi_gdof[l][k] * dot_prod;
					}
				}

				// normalize residual vector
				double rotnorm = 0.0;
				for( int l = 0; l < size; l++ ) {
					rotnorm += Qi_gdof[l][j] * Qi_gdof[l][j];
				}

				rotnorm = 1.0 / sqrt( rotnorm );

				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][j] = Qi_gdof[l][j] * rotnorm;
				}
			}

			// orthogonalize original eigenvectors against gdof
			// number of evec that survive orthogonalization
			int curr_evec = ConservedDegreesOfFreedom;
			for( int j = 0; j < size; j++ ) { // <-- vector we're orthogonalizing
				// to match ProtoMol we only include size instead of size + cdof vectors
				// Note: for every vector that is skipped due to a low norm,
				// we add an additional vector to replace it, so we could actually
				// use all size original eigenvectors
				if( curr_evec == size ) {
					break;
				}

				// orthogonalize original eigenvectors in order from smallest magnitude
				// eigenvalue to biggest
				int col = sortedEvalPairs.at( j ).second;

				// copy original vector to Qi_gdof -- updated in place
				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][curr_evec] = Qi[l][col];
				}

				// get dot products with previous vectors
				for( int k = 0; k < curr_evec; k++ ) { // <-- vector orthog against
					// dot product between original vector and previously
					// orthogonalized vectors
					double dot_prod = 0.0;
					for( int l = 0; l < size; l++ ) {
						dot_prod += Qi_gdof[l][k] * Qi[l][col];
					}

					// subtract from current vector -- update in place
					for( int l = 0; l < size; l++ ) {
						Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] - Qi_gdof[l][k] * dot_prod;
					}
				}

				//normalize residual vector
				double norm = 0.0;
				for( int l = 0; l < size; l++ ) {
					norm += Qi_gdof[l][curr_evec] * Qi_gdof[l][curr_evec];
				}

				// if norm less than 1/20th of original
				// continue on to next vector
				// we don't update curr_evec so this vector
				// will be overwritten
				if( norm < 0.05 ) {
					continue;
				}

				// scale vector
				norm = sqrt( norm );
				for( int l = 0; l < size; l++ ) {
					Qi_gdof[l][curr_evec] = Qi_gdof[l][curr_evec] / norm;
				}

				curr_evec++;
			}
			
			// 4. Copy eigenpairs to big array
			//    This is necessary because we have to sort them, and determine
			//    the cutoff eigenvalue for everybody.
			// we assume curr_evec <= size
			for( int j = 0; j < curr_evec; j++ ) {
				int col = sortedEvalPairs.at( j ).second;
				eval[startatom + j] = di[col];

				// orthogonalized eigenvectors already sorted by eigenvalue
				for( int k = 0; k < size; k++ ) {
					evec[startatom + k][startatom + j] = Qi_gdof[k][j];
				}
			}
		}
	}
}
