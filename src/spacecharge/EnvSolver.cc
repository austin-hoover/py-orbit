#include "EnvSolver.hh"


EnvSolver::EnvSolver(double perveance): CppPyWrapper(NULL)
{
    Q = perveance;
}


void EnvSolver::trackBunch(Bunch* bunch, double length)
{
    // Compute ellipse size and orientation
    a = bunch->x(0); e = bunch->y(0);
    b = bunch->x(1); f = bunch->y(1);
    double xx = a*a + b*b; // 4 * <x^2>
    double yy = e*e + f*f; // 4 * <y^2>
    double xy = a*e + b*f; // 4 * <xy>
    phi = -0.5 * atan2(2*xy, xx - yy);
    double _cos = cos(phi);
    double _sin = sin(phi);
    double cos2 = _cos * _cos;
    double sin2 = _sin * _sin;
    double sincos = _sin * _cos;
    double cx = sqrt(abs(xx*cos2 + yy*sin2 - 2*xy*sincos));
    double cy = sqrt(abs(xx*sin2 + yy*cos2 + 2*xy*sincos));
    double factor = length * (2 * Q / (cx + cy));
        
    // Track envelope
    if (cx > 0) {
        bunch->xp(0) += factor * (a*cos2 - e*sincos)/cx;
        bunch->xp(1) += factor * (b*cos2 - f*sincos)/cx;
        bunch->yp(0) += factor * (e*sin2 - a*sincos)/cx;
        bunch->yp(1) += factor * (f*sin2 - b*sincos)/cx;
    }
    if (cy > 0) {
        bunch->xp(0) += factor * (a*sin2 + e*sincos)/cy;
        bunch->xp(1) += factor * (b*sin2 + f*sincos)/cy;
        bunch->yp(0) += factor * (e*cos2 + a*sincos)/cy;
        bunch->yp(1) += factor * (f*cos2 + b*sincos)/cy;
    }
    
    // Track bunch particles
    //     To do:
    //        * Double check formula for electric field outside envelope
    //        * Add conducting boundary (Jeff's code)
    double cx2 = cx * cx;
    double cy2 = cy * cy;
    double x, y, xn, yn, x2, y2, xn2, yn2;
    double t1, B, C, D;
    double delta_xpn, delta_ypn;
    bool in_ellipse;
    
    for (int i = 2; i < bunch->getSize(); i++) {
        delta_xpn = 0.0;
        delta_ypn = 0.0;
        x = bunch->x(i);
        y = bunch->y(i);
        xn = x * _cos - y * _sin;
        yn = x * _sin + y * _cos;
        x2 = x * x;
        y2 = y * y;
        xn2 = xn * xn;
        yn2 = yn * yn;
        in_ellipse = xn2/cx2 + yn2/cy2 <= 1.0;
        if (in_ellipse) {
            if (cx > 0.0) {
                delta_xpn = factor * xn / cx;
            }
            if (cy > 0.0) {
                delta_ypn = factor * yn / cy;
            }
        }
        else {
            // Using expression derived here: https://arxiv.org/abs/physics/0108040. This
            // has not been tested yet!
            B = xn2 + yn2 - cx2 - cy2;
            C = xn2*cy2 + yn2*cx2 - cx2*cy2;
            t1 = pow(B*B/4 + C, 0.5) + B/2;
            D = pow((cx2 + t1) * (cy2 + t1), 0.5);
            delta_xpn = 2 * Q * xn / ((cx2 + t1) + D);
            delta_ypn = 2 * Q * yn / ((cy2 + t1) + D);
        }
        bunch->xp(i) += +delta_xpn * _cos + delta_ypn * _sin;
        bunch->yp(i) += -delta_xpn * _sin + delta_ypn * _cos;
    }
}
