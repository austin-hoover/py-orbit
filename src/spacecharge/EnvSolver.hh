/** Envelope solver for uniform density elliptical beam.
 
 The method uses the first two particles in the bunch to track the beam envelope, which
 is then used to apply linear space charge kicks to the other particles in the bunch. The
 boundary of a tilted ellipse can be parameterized as
 The boundary of a tilted ellipse in real space can be parameterized as:
     x = a*cos(psi) + b*sin(psi),
     y = e*cos(psi) + f*sin(psi),
 where 0 <= psi <= 2pi.
 
 Reference: 
 V. Danilov, S. Cousineau, S. Henderson, and J. Holmes, Self-consistent time dependent
 two dimensional and three dimensional space charge distributions with linear force,
 Physical Review Special Topics - Accelerators and Beams 6, 74–85 (2003).
*/

#ifndef ENVSOLVER_H
#define ENVSOLVER_H

#include "Bunch.hh"
#include "CppPyWrapper.hh"

using namespace std;

class EnvSolver: public OrbitUtils::CppPyWrapper
{
    public:
    
    double Q; // beam perveance

    /** Constructor */
    EnvSolver(double perveance);

    /** Apply space charge kick.
     
     The method tracks two particles. The particle coordinates map to the parameters {a, b, e, f} and their
     slopes in the following way:
         particle 1 --> {x:a, x':a', y:e, y':e'},
         particle 2 --> {x:e, x':e', y:f, y':f'}.
    */
    void trackBunch(Bunch* bunch, double length);
};

#endif
