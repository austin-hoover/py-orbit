#ifndef WRAP_DANILOVENVSOLVER_H
#define WRAP_DANILOVENVSOLVER_H

#include "Python.h"

#ifdef __cplusplus
extern "C" {
#endif

  namespace wrap_spacecharge{
    void initDanilovEnvSolver(PyObject* module);
  }
    
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // WRAP_ENVSOLVER_H
