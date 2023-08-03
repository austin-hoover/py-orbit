#ifndef DANILOVENVELOPESOLVER20_H
#define DANILOVENVELOPESOLVER20_H

#include "Bunch.hh"
#include "CppPyWrapper.hh"

using namespace std;


class DanilovEnvelopeSolver20: public OrbitUtils::CppPyWrapper {
    public:
        DanilovEnvelopeSolver20(double perveance, double _eps_x, double _eps_y);
    
        void trackBunch(Bunch* bunch, double length);
    
        void setPerveance(double perveance);
    
        void setEmittance(double _eps_x, double _eps_y);

    private:
        double Q;  // beam perveance    
        double eps_x;  // (4 * sqrt(<xx><x'x'> - <xx'><xx'>))
        double eps_y;  // (4 * sqrt(<yy><y'y'> - <yy'><yy'>))
};

#endif