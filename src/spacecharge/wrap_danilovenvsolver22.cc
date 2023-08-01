#include "orbit_mpi.hh"
#include "pyORBIT_Object.hh"
#
#include "wrap_danilovenvsolver22.hh"
#include "wrap_spacecharge.hh"
#include "wrap_bunch.hh"

#include <iostream>

#include "DanilovEnvSolver22.hh"

using namespace OrbitUtils;

namespace wrap_spacecharge{

#ifdef __cplusplus
extern "C" {
#endif

    //---------------------------------------------------------
    //Python DanilovEnvSolver22 class definition
    //---------------------------------------------------------

    //constructor for python class wrapping DanilovEnvSolver22 instance
    //It never will be called directly

    static PyObject* DanilovEnvSolver22_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
        pyORBIT_Object* self;
        self = (pyORBIT_Object *) type->tp_alloc(type, 0);
        self->cpp_obj = NULL;
        //std::cerr<<"The DanilovEnvSolver22 new has been called!"<<std::endl;
        return (PyObject *) self;
    }
    
  //initializator for python DanilovEnvSolver22 class
  //this is implementation of the __init__ method DanilovEnvSolver22(double perveance)
  static int DanilovEnvSolver22_init(pyORBIT_Object *self, PyObject *args, PyObject *kwds)
  {
      double ex, ey, Q;
        if (!PyArg_ParseTuple(args, "d:__init__", &Q)) {
            ORBIT_MPI_Finalize("PyDanilovEnvSolver22 - DanilovEnvSolver22(Q) - constructor needs parameters.");
        }
      self->cpp_obj = new DanilovEnvSolver22(Q);
      ((DanilovEnvSolver22*) self->cpp_obj)->setPyWrapper((PyObject*) self);
      //std::cerr<<"The DanilovEnvSolver22 __init__ has been called!"<<std::endl;
      return 0;
    }
    
    //trackBunch(Bunch* bunch, double length)
    static PyObject* DanilovEnvSolver22_trackBunch(PyObject *self, PyObject *args) {
        int nVars = PyTuple_Size(args);
        pyORBIT_Object* pyDanilovEnvSolver22 = (pyORBIT_Object*) self;
        DanilovEnvSolver22* cpp_DanilovEnvSolver22 = (DanilovEnvSolver22*) pyDanilovEnvSolver22->cpp_obj;
        PyObject* pyBunch;
        double length;
        
        if (!PyArg_ParseTuple(args, "Od:trackBunch", &pyBunch, &length)){
            ORBIT_MPI_Finalize("PyDanilovEnvSolver22.trackBunch(pyBunch, length) - method needs parameters.");
        }
        PyObject* pyORBIT_Bunch_Type = wrap_orbit_bunch::getBunchType("Bunch");
        if (!PyObject_IsInstance(pyBunch,pyORBIT_Bunch_Type)){
            ORBIT_MPI_Finalize("PyDanilovEnvSolver22.trackBunch(pyBunch,length) - pyBunch is not Bunch.");
        }
        Bunch* cpp_bunch = (Bunch*) ((pyORBIT_Object*)pyBunch)->cpp_obj;
        cpp_DanilovEnvSolver22->trackBunch(cpp_bunch, length);
        Py_INCREF(Py_None);
        return Py_None;
  }

  //-----------------------------------------------------
  // Destructor for python DanilovEnvSolver22 class (__del__ method).
  //-----------------------------------------------------
  static void DanilovEnvSolver22_del(pyORBIT_Object* self){
        DanilovEnvSolver22* cpp_DanilovEnvSolver22 = (DanilovEnvSolver22*) self->cpp_obj;
        if(cpp_DanilovEnvSolver22 != NULL){
            delete cpp_DanilovEnvSolver22;
        }
        self->ob_type->tp_free((PyObject*)self);
  }
  
  // Definition of the methods of the python DanilovEnvSolver22 wrapper class
  // they will be vailable from python level
  static PyMethodDef DanilovEnvSolver22ClassMethods[] = {
        { "trackBunch",  DanilovEnvSolver22_trackBunch, METH_VARARGS,"track the bunch - trackBunch(pyBunch, length)"},
        {NULL}
  };
  
  // Definition of the memebers of the python DanilovEnvSolver22 wrapper class
  // they will be vailable from python level
  static PyMemberDef DanilovEnvSolver22ClassMembers [] = {
        {NULL}
  };

    // New python DanilovEnvSolver22 wrapper type definition
    static PyTypeObject pyORBIT_DanilovEnvSolver22_Type = {
        PyObject_HEAD_INIT(NULL)
        0, /*ob_size*/
        "DanilovEnvSolver22", /*tp_name*/
        sizeof(pyORBIT_Object), /*tp_basicsize*/
        0, /*tp_itemsize*/
        (destructor) DanilovEnvSolver22_del , /*tp_dealloc*/
        0, /*tp_print*/
        0, /*tp_getattr*/
        0, /*tp_setattr*/
        0, /*tp_compare*/
        0, /*tp_repr*/
        0, /*tp_as_number*/
        0, /*tp_as_sequence*/
        0, /*tp_as_mapping*/
        0, /*tp_hash */
        0, /*tp_call*/
        0, /*tp_str*/
        0, /*tp_getattro*/
        0, /*tp_setattro*/
        0, /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "The DanilovEnvSolver22 python wrapper", /* tp_doc */
        0, /* tp_traverse */
        0, /* tp_clear */
        0, /* tp_richcompare */
        0, /* tp_weaklistoffset */
        0, /* tp_iter */
        0, /* tp_iternext */
        DanilovEnvSolver22ClassMethods, /* tp_methods */
        DanilovEnvSolver22ClassMembers, /* tp_members */
        0, /* tp_getset */
        0, /* tp_base */
        0, /* tp_dict */
        0, /* tp_descr_get */
        0, /* tp_descr_set */
        0, /* tp_dictoffset */
        (initproc) DanilovEnvSolver22_init, /* tp_init */
        0, /* tp_alloc */
        DanilovEnvSolver22_new, /* tp_new */
    };

    //--------------------------------------------------
    // Initialization function of the pyDanilovEnvSolver22 class
    // It will be called from SpaceCharge wrapper initialization
    //--------------------------------------------------
  void initDanilovEnvSolver22(PyObject* module){
        if (PyType_Ready(&pyORBIT_DanilovEnvSolver22_Type) < 0) return;
        Py_INCREF(&pyORBIT_DanilovEnvSolver22_Type);
        PyModule_AddObject(module, "DanilovEnvSolver22", (PyObject *)&pyORBIT_DanilovEnvSolver22_Type);
    }

#ifdef __cplusplus
}
#endif

//end of namespace wrap_spacecharge
}
