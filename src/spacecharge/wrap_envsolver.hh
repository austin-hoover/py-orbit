#ifndef WRAP_ENVSOLVER_H
#define WRAP_ENVSOLVER_H

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

  namespace wrap_spacecharge{
    void initEnvSolver(PyObject* module);
  }
    
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // WRAP_ENVSOLVER_H
