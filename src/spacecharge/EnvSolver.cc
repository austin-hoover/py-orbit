#include "EnvSolver.hh"


EnvSolver::EnvSolver(double perveance): CppPyWrapper(NULL)
{
    Q = perveance;
}


void EnvSolver::trackBunch(Bunch* bunch, double length)
{
    double xx, yy, xy, _cos, _sin, cos2, sin2, sincos, factor;
    // Compute ellipse size and orientation
    a = bunch->x(0); e = bunch->y(0);
    b = bunch->x(1); f = bunch->y(1);
    xx = a*a + b*b; // 4 * <x^2>
    yy = e*e + f*f; // 4 * <y^2>
    xy = a*e + b*f; // 4 * <xy>
    phi = -0.5 * atan2(2*xy, xx - yy);
    _cos = cos(phi);
    _sin = sin(phi);
    cos2 = _cos * _cos;
    sin2 = _sin * _sin;
    sincos = _sin * _cos;
    cx = sqrt(abs(xx*cos2 + yy*sin2 - 2*xy*sincos));
    cy = sqrt(abs(xx*sin2 + yy*cos2 + 2*xy*sincos));
    factor = 2 * Q / (cx + cy);
    
    // Track envelope
    if (cx > 0) {
        bunch->xp(0) += (factor * (a*cos2 - e*sincos)/cx) * length;
        bunch->xp(1) += (factor * (b*cos2 - f*sincos)/cx) * length;
        bunch->yp(0) += (factor * (e*sin2 - a*sincos)/cx) * length;
        bunch->yp(1) += (factor * (f*sin2 - b*sincos)/cx) * length;
    }
    if (cy > 0) {
        bunch->xp(0) += (factor * (a*sin2 + e*sincos)/cy) * length;
        bunch->xp(1) += (factor * (b*sin2 + f*sincos)/cy) * length;
        bunch->yp(0) += (factor * (e*cos2 + a*sincos)/cy) * length;
        bunch->yp(1) += (factor * (f*cos2 + b*sincos)/cy) * length;
    }
    
    // Track bunch particles
    //   To do: add Jeff's code to handle particles outside ellipse
    double x, y;
    for (int i = 2; i < bunch->getSize(); i++) {
        x = bunch->x(i); 
        y = bunch->y(i);
        if (cx > 0) {
            bunch->xp(i) += (factor * (cos2*x - sincos*y)/cx) * length;
            bunch->yp(i) += (factor * (sin2*y - sincos*x)/cx) * length;
        }
        if (cy > 0) {
            bunch->xp(i) += (factor * (sin2*x + sincos*y)/cy) * length;
            bunch->yp(i) += (factor * (cos2*y + sincos*x)/cy) * length;
        }
    }
}
