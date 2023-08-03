#include "orbit_mpi.hh"
#include "pyORBIT_Object.hh"
#
#include "wrap_danilov_envelope_solver_20.hh"
#include "wrap_envelope.hh"
#include "wrap_bunch.hh"

#include <iostream>

#include "DanilovEnvelopeSolver20.hh"

using namespace OrbitUtils;

namespace wrap_envelope{

#ifdef __cplusplus
extern "C" {
#endif

    // ---------------------------------------------------------
    // Python DanilovEnvelopeSolver20 class definition
    // ---------------------------------------------------------

    // Constructor for python class wrapping DanilovEnvelopeSolver20 instance.
    // It never will be called directly.
    static PyObject* DanilovEnvelopeSolver20_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
        pyORBIT_Object* self;
        self = (pyORBIT_Object *) type->tp_alloc(type, 0);
        self->cpp_obj = NULL;
        //std::cerr<<"The DanilovEnvelopeSolver20 new has been called!"<<std::endl;
        return (PyObject *) self;
    }
    
    // Initializator for python DanilovEnvelopeSolver20 class.
    // This is implementation of the __init__ method DanilovEnvelopeSolver20(double perveance, double eps_x, double eps_y).
    static int DanilovEnvelopeSolver20_init(pyORBIT_Object *self, PyObject *args, PyObject *kwds) {
        double perveance, _eps_x, _eps_y;
        if (!PyArg_ParseTuple(args, "ddd:__init__", &perveance, &_eps_x, &_eps_y)) {
            ORBIT_MPI_Finalize("PyDanilovEnvelopeSolver20 - DanilovEnvelopeSolver20(perveance, eps_x, eps_y) - constructor needs parameters.");
        }
        self->cpp_obj = new DanilovEnvelopeSolver20(perveance, _eps_x, _eps_y);
        ((DanilovEnvelopeSolver20*) self->cpp_obj)->setPyWrapper((PyObject*) self);
        //std::cerr<<"The DanilovEnvelopeSolver20 __init__ has been called!"<<std::endl;
        return 0;
    }
    
    // trackBunch(Bunch* bunch, double length)
    static PyObject* DanilovEnvelopeSolver20_trackBunch(PyObject *self, PyObject *args) {
        int nVars = PyTuple_Size(args);
        pyORBIT_Object* pyDanilovEnvelopeSolver20 = (pyORBIT_Object*) self;
        DanilovEnvelopeSolver20* cpp_DanilovEnvelopeSolver20 = (DanilovEnvelopeSolver20*) pyDanilovEnvelopeSolver20->cpp_obj;
        PyObject* pyBunch;
        double length;
        
        if (!PyArg_ParseTuple(args, "Od:trackBunch", &pyBunch, &length)){
            ORBIT_MPI_Finalize("PyDanilovEnvelopeSolver20.trackBunch(pyBunch, length) - method needs parameters.");
        }
        PyObject* pyORBIT_Bunch_Type = wrap_orbit_bunch::getBunchType("Bunch");
        if (!PyObject_IsInstance(pyBunch,pyORBIT_Bunch_Type)){
            ORBIT_MPI_Finalize("PyDanilovEnvelopeSolver20.trackBunch(pyBunch,length) - pyBunch is not Bunch.");
        }
        Bunch* cpp_bunch = (Bunch*) ((pyORBIT_Object*)pyBunch)->cpp_obj;
        cpp_DanilovEnvelopeSolver20->trackBunch(cpp_bunch, length);
        Py_INCREF(Py_None);
        return Py_None;
  }
    
    // setPerveance(double perveance)
    static PyObject* DanilovEnvelopeSolver20_setPerveance(PyObject *self, PyObject *args) {
        int nVars = PyTuple_Size(args);
        pyORBIT_Object* pyDanilovEnvelopeSolver20 = (pyORBIT_Object*) self;
        DanilovEnvelopeSolver20* cpp_DanilovEnvelopeSolver20 = (DanilovEnvelopeSolver20*) pyDanilovEnvelopeSolver20->cpp_obj;
        double perveance;
        
        if (!PyArg_ParseTuple(args, "d:setPerveance", &perveance)){
            ORBIT_MPI_Finalize("PyDanilovEnvelopeSolver20.setPerveance(perveance) - method needs parameters.");
        }
        cpp_DanilovEnvelopeSolver20->setPerveance(perveance);
        Py_INCREF(Py_None);
        return Py_None;
    }
        
    // setEmittance(double eps_x, double eps_y)
    static PyObject* DanilovEnvelopeSolver20_setEmittance(PyObject *self, PyObject *args) {
        int nVars = PyTuple_Size(args);
        pyORBIT_Object* pyDanilovEnvelopeSolver20 = (pyORBIT_Object*) self;
        DanilovEnvelopeSolver20* cpp_DanilovEnvelopeSolver20 = (DanilovEnvelopeSolver20*) pyDanilovEnvelopeSolver20->cpp_obj;
        double eps_x;
        double eps_y;
        
        if (!PyArg_ParseTuple(args, "dd:setEmittance", &eps_x, &eps_y)){
            ORBIT_MPI_Finalize("PyDanilovEnvelopeSolver20.setEmittance(eps_x, eps_y) - method needs parameters.");
        }
        cpp_DanilovEnvelopeSolver20->setEmittance(eps_x, eps_y);
        Py_INCREF(Py_None);
        return Py_None;
    }


  //-----------------------------------------------------
  // Destructor for python DanilovEnvelopeSolver20 class (__del__ method).
  //-----------------------------------------------------
  static void DanilovEnvelopeSolver20_del(pyORBIT_Object* self){
        DanilovEnvelopeSolver20* cpp_DanilovEnvelopeSolver20 = (DanilovEnvelopeSolver20*) self->cpp_obj;
        if(cpp_DanilovEnvelopeSolver20 != NULL){
            delete cpp_DanilovEnvelopeSolver20;
        }
        self->ob_type->tp_free((PyObject*)self);
  }
  
  // Definition of the methods of the python DanilovEnvelopeSolver20 wrapper class
  // they will be vailable from python level
  static PyMethodDef DanilovEnvelopeSolver20ClassMethods[] = {
        { "trackBunch",  DanilovEnvelopeSolver20_trackBunch, METH_VARARGS,"track the bunch - trackBunch(pyBunch, length)"},
        { "setPerveance",  DanilovEnvelopeSolver20_setPerveance, METH_VARARGS,"setPerveance(length)"},
        { "setEmittance",  DanilovEnvelopeSolver20_setEmittance, METH_VARARGS,"setEmittance(eps_x, eps_y)"},
        {NULL}
  };
  
  // Definition of the memebers of the python DanilovEnvelopeSolver20 wrapper class
  // they will be vailable from python level
  static PyMemberDef DanilovEnvelopeSolver20ClassMembers [] = {
        {NULL}
  };

    // New python DanilovEnvelopeSolver20 wrapper type definition
    static PyTypeObject pyORBIT_DanilovEnvelopeSolver20_Type = {
        PyObject_HEAD_INIT(NULL)
        0, /*ob_size*/
        "DanilovEnvelopeSolver20", /*tp_name*/
        sizeof(pyORBIT_Object), /*tp_basicsize*/
        0, /*tp_itemsize*/
        (destructor) DanilovEnvelopeSolver20_del , /*tp_dealloc*/
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
        "The DanilovEnvelopeSolver20 python wrapper", /* tp_doc */
        0, /* tp_traverse */
        0, /* tp_clear */
        0, /* tp_richcompare */
        0, /* tp_weaklistoffset */
        0, /* tp_iter */
        0, /* tp_iternext */
        DanilovEnvelopeSolver20ClassMethods, /* tp_methods */
        DanilovEnvelopeSolver20ClassMembers, /* tp_members */
        0, /* tp_getset */
        0, /* tp_base */
        0, /* tp_dict */
        0, /* tp_descr_get */
        0, /* tp_descr_set */
        0, /* tp_dictoffset */
        (initproc) DanilovEnvelopeSolver20_init, /* tp_init */
        0, /* tp_alloc */
        DanilovEnvelopeSolver20_new, /* tp_new */
    };

    //--------------------------------------------------
    // Initialization function of the pyDanilovEnvelopeSolver20 class
    // It will be called from SpaceCharge wrapper initialization
    //--------------------------------------------------
  void initDanilovEnvelopeSolver20(PyObject* module){
        if (PyType_Ready(&pyORBIT_DanilovEnvelopeSolver20_Type) < 0) return;
        Py_INCREF(&pyORBIT_DanilovEnvelopeSolver20_Type);
        PyModule_AddObject(module, "DanilovEnvelopeSolver20", (PyObject *)&pyORBIT_DanilovEnvelopeSolver20_Type);
    }

#ifdef __cplusplus
}
#endif

//end of namespace wrap_spacecharge
}
