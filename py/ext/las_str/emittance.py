#-----------------------------------------------------
#Track bunch with r and p through the external field 
# The field is 1 T and has direction (0,1,0)
#-----------------------------------------------------

import sys,math,os,orbit_mpi,random

from bunch import *
from trackerrk4 import *
from laserstripping import *
from orbit_utils import *

from orbit_mpi import mpi_comm,mpi_datatype,mpi_op


def xyBeamEmittances(bunch):
    
    com = bunch.getMPIComm()
    mpi_size = orbit_mpi.MPI_Comm_size(com)
    op = mpi_op.MPI_SUM
    data_type = mpi_datatype.MPI_DOUBLE
    
    
    N_part_loc = bunch.getSize()
    N_part_glob = bunch.getSizeGlobal()
    
    P = 0
    for i in range(N_part_loc):
        P += bunch.pz(i)
    P = orbit_mpi.MPI_Allreduce(P,data_type,op,com)
    P = P/N_part_glob
    
    XP0 = 0
    YP0 = 0
    X0 = 0
    Y0 = 0
    for i in range(N_part_loc):
        XP0 += bunch.px(i)/bunch.pz(i)
        YP0 += bunch.py(i)/bunch.pz(i)
        X0 += bunch.x(i)
        Y0 += bunch.y(i)
    XP0 = orbit_mpi.MPI_Allreduce(XP0,data_type,op,com)
    YP0 = orbit_mpi.MPI_Allreduce(YP0,data_type,op,com)
    X0 = orbit_mpi.MPI_Allreduce(X0,data_type,op,com)
    Y0 = orbit_mpi.MPI_Allreduce(Y0,data_type,op,com)
    XP0 = XP0/N_part_glob
    YP0 = YP0/N_part_glob
    X0 = X0/N_part_glob
    Y0 = Y0/N_part_glob
 

    
    XP2 = 0
    YP2 = 0
    X2 = 0
    Y2 = 0
    PXP = 0
    PYP = 0
    for i in range(N_part_loc):
        
        XP = bunch.px(i)/bunch.pz(i) - XP0
        YP = bunch.py(i)/bunch.pz(i) - YP0
        X = bunch.x(i) - X0
        Y = bunch.y(i) - Y0
        
        XP2 += XP*XP
        YP2 += YP*YP
        X2 += X*X
        Y2 += Y*Y
        PXP += XP*X
        PYP += YP*Y
        
    XP2 = orbit_mpi.MPI_Allreduce(XP2,data_type,op,com)
    YP2 = orbit_mpi.MPI_Allreduce(YP2,data_type,op,com)
    X2 = orbit_mpi.MPI_Allreduce(X2,data_type,op,com)
    Y2 = orbit_mpi.MPI_Allreduce(Y2,data_type,op,com)
    PXP = orbit_mpi.MPI_Allreduce(PXP,data_type,op,com)
    PYP = orbit_mpi.MPI_Allreduce(PYP,data_type,op,com)
    XP2 = XP2/N_part_glob
    YP2 = YP2/N_part_glob
    X2 = X2/N_part_glob
    Y2 = Y2/N_part_glob
    PXP = PXP/N_part_glob
    PYP = PYP/N_part_glob
    
    
    ex = math.sqrt(X2*XP2 - PXP*PXP)
    ey = math.sqrt(Y2*YP2 - PYP*PYP)
    
    E = math.sqrt(P*P + bunch.mass()*bunch.mass())
    
    beta_rel = P/E
    gamma_rel = 1./math.sqrt(1 - beta_rel*beta_rel) 
    
    
    exn = ex*beta_rel*gamma_rel
    eyn = ey*beta_rel*gamma_rel
    print beta_rel*gamma_rel

    return exn, eyn

 


def BeamEmittances(bunch,beta):
    
    com = bunch.getMPIComm()
    mpi_size = orbit_mpi.MPI_Comm_size(com)
    op = mpi_op.MPI_SUM
    data_type = mpi_datatype.MPI_DOUBLE
    
    
    N_part_loc = bunch.getSize()
    N_part_glob = bunch.getSizeGlobal()
    
    P = 0
    for i in range(N_part_loc):
        P += bunch.pz(i)
    P = orbit_mpi.MPI_Allreduce(P,data_type,op,com)
    P = P/N_part_glob
    
    XP0 = 0
    YP0 = 0

    for i in range(N_part_loc):
        XP0 += bunch.px(i)/bunch.pz(i)
        YP0 += bunch.py(i)/bunch.pz(i)
        
    XP0 = orbit_mpi.MPI_Allreduce(XP0,data_type,op,com)
    YP0 = orbit_mpi.MPI_Allreduce(YP0,data_type,op,com)
    XP0 = XP0/N_part_glob
    YP0 = YP0/N_part_glob

 

    
    XP2 = 0
    YP2 = 0

    for i in range(N_part_loc):
        
        XP = bunch.px(i)/bunch.pz(i) - XP0
        YP = bunch.py(i)/bunch.pz(i) - YP0

        
        XP2 += XP*XP
        YP2 += YP*YP

        
    XP2 = orbit_mpi.MPI_Allreduce(XP2,data_type,op,com)
    YP2 = orbit_mpi.MPI_Allreduce(YP2,data_type,op,com)

    XP2 = XP2/N_part_glob
    YP2 = YP2/N_part_glob

    
    
    ex = XP2*beta
    ey = YP2*beta
    
    E = math.sqrt(P*P + bunch.mass()*bunch.mass())
    
    beta_rel = P/E
    gamma_rel = 1./math.sqrt(1 - beta_rel*beta_rel) 
    
    
    exn = ex*beta_rel*gamma_rel
    eyn = ey*beta_rel*gamma_rel

    return exn, eyn 


def Freq_spread(bunch,la,n):
    
    delta_E = 1./2. - 1./(2.*n*n)
    m = bunch.mass()
    
    op = mpi_op.MPI_SUM
    data_type = mpi_datatype.MPI_DOUBLE
    mpi_size = orbit_mpi.MPI_Comm_size(mpi_comm.MPI_COMM_WORLD)
    
    N = bunch.getSize()
    pz = 0
    for i in range(N):
        pz += bunch.pz(i)
    pz = orbit_mpi.MPI_Allreduce(pz,data_type,op,mpi_comm.MPI_COMM_WORLD)
    pz = pz/(mpi_size*N)
    
    
    E = math.sqrt(pz*pz + m*m)
    TK = E - m

    
    la0 = 2*math.pi*5.291772108e-11/7.297352570e-3/delta_E
    te = TK - m*(la/la0-1)
    kzz = te/math.sqrt(pz*pz-te*te)

    kx = -1.
    ky = 0.
    kz = kzz
    
    
    om = 0
    om2 = 0
    
    for i in range(N):
        
        px = bunch.px(i)
        py = bunch.py(i)
        pz = bunch.pz(i)
        
        P2 = px*px + py*py + pz*pz
        K = math.sqrt(kx*kx + ky*ky + kz*kz)
        P = math.sqrt(P2) 
        
        E = math.sqrt(P2 + m*m)
        
        beta = P/E
        gamma = E/m
        
        cos = (px*kz + py*ky + pz*kz)/(K*P)
        
        la0 = la/(gamma*(1-beta*cos))
        omega = 2*math.pi*5.291772108e-11/7.297352570e-3/la0
        
        om += omega
        om2 += omega*omega


#        f = open('bunch_parameters.txt','a')
#        print >>f, omega
#        f.close()
        
    om = orbit_mpi.MPI_Allreduce(om,data_type,op,mpi_comm.MPI_COMM_WORLD)
    om = om/(mpi_size*N)     
    
    om2 = orbit_mpi.MPI_Allreduce(om2,data_type,op,mpi_comm.MPI_COMM_WORLD)        
    om2 = om2/(mpi_size*N)
    
    sigma_om = math.sqrt(om2 - om**2)    

    return (om, sigma_om)
        

