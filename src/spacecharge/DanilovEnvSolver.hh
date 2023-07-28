/** Envelope solver for the {2, 2} Danilov distribution.
 
 The boundary of the tilted ellipse in real-space can be parameterized as:
     x = a*cos(psi) + b*sin(psi),
     y = e*cos(psi) + f*sin(psi),
 where 0 <= psi <= 2pi. The method uses the first two bunch particles in the 
 bunch to track the beam envelope, which is then used to apply linear space 
 charge kicks to the other particles in the bunch. 
  
 Reference: 
 V. Danilov, S. Cousineau, S. Henderson, and J. Holmes, Self-consistent time dependent
 two dimensional and three dimensional space charge distributions with linear force,
 Physical Review Special Topics - Accelerators and Beams 6, 74–85 (2003).
*/

#ifndef DANILOVENVSOLVER_H
#define DANILOVENVSOLVER_H

#include "Bunch.hh"
#include "CppPyWrapper.hh"

using namespace std;

class DanilovEnvSolver: public OrbitUtils::CppPyWrapper {
    public:
        DanilovEnvSolver(double perveance);
        void trackBunch(Bunch* bunch, double length);

    private:
        double Q;  // beam perveance
        double a, b, e, f;  // parameters of ellipse in real space
        double phi;  // tilt angle below x axis
        double cx, cy;  // ellipse radii
};

#endif