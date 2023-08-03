#include "orbit_mpi.hh"

#include "wrap_envelope.hh"
#include "wrap_danilov_envelope_solver_20.hh"
#include "wrap_danilov_envelope_solver_22.hh"


static PyMethodDef envelopeMethods[] = {{NULL, NULL}};

#ifdef __cplusplus
extern "C" {
#endif

namespace wrap_envelope {
    
    void initenvelope(){
        PyObject* module = Py_InitModule("envelope", envelopeMethods);
        wrap_envelope::initDanilovEnvelopeSolver20(module);
        wrap_envelope::initDanilovEnvelopeSolver22(module);
    }

    PyObject* getEnvelopeType(const char* name){
        PyObject* mod = PyImport_ImportModule("envelope");
        PyObject* pyType = PyObject_GetAttrString(mod,name);
        Py_DECREF(mod);
        Py_DECREF(pyType);
        return pyType;
    }
}
    

#ifdef __cplusplus
}
#endif