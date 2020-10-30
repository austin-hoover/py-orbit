#include "EnvSolverDanilov.hh"

EnvSolverDanilov::EnvSolverDanilov(double perveance): CppPyWrapper(NULL)
{
    Q = perveance;
}
    
void EnvSolverDanilov::trackBunch(Bunch* bunch, double length)
{
    double phi, cosp, sinp, cos2p, sin2p;
    double cx, cy, factor;
    double a, b, e, f;
    double x, y;

    // Compute ellipse size and orientation
    a = bunch->x(0);
    b = bunch->x(1);
    e = bunch->y(0);
    f = bunch->y(1);

    phi = -0.5 * atan2(2*(a*e + b*f), (a*a + b*b - e*e - f*f));
    cosp = cos(phi);
    sinp = sin(phi);
    cos2p = cosp*cosp;
    sin2p = sinp*sinp;

    cx = sqrt(pow(a*f-b*e, 2) / ((e*e+f*f)*cos2p + (a*a+b*b)*sin2p +  2*(a*e+b*f)*cosp*sinp));
    cy = sqrt(pow(a*f-b*e, 2) / ((a*a+b*b)*cos2p + (e*e+f*f)*sin2p -  2*(a*e+b*f)*cosp*sinp));
    factor = 2 * Q / (cx + cy);

    // Track the envelope parameters
    bunch->xp(0) += (factor * ((a*cos2p - e*sinp*cosp)/cx + (a*sin2p + e*sinp*cosp)/cy)) * length;
    bunch->xp(1) += (factor * ((b*cos2p - f*sinp*cosp)/cx + (b*sin2p + f*sinp*cosp)/cy)) * length;
    bunch->yp(0) += (factor * ((e*cos2p + a*sinp*cosp)/cy + (e*sin2p - a*sinp*cosp)/cx)) * length;
    bunch->yp(1) += (factor * ((f*cos2p + b*sinp*cosp)/cy + (f*sin2p - b*sinp*cosp)/cx)) * length;
    
    // Track the test bunch particles
    for (int i = 2; i < bunch->getSize(); i++) {
        x = bunch->x(i);
        y = bunch->y(i);
        bunch->xp(i) += (factor * ((cos2p/cx + sin2p/cy)*x + (1/cy - 1/cx)*sinp*cosp*y)) * length;
        bunch->yp(i) += (factor * ((sin2p/cx + cos2p/cy)*y + (1/cy - 1/cx)*sinp*cosp*x)) * length;
    }
}
