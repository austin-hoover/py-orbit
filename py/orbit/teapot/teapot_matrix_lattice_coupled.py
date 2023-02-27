import math
import os

from bunch import Bunch
from orbit.lattice import AccActionsContainer
from orbit.lattice import AccLattice
from orbit.lattice import AccNode
from orbit.matrix_lattice import BaseMATRIX
from orbit.matrix_lattice import MATRIX_Lattice_Coupled
from orbit.teapot import BaseTEAPOT
from orbit.teapot import RingRFTEAPOT
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot_base import MatrixGenerator
from orbit.utils import orbitFinalize


class TEAPOT_MATRIX_Lattice_Coupled(MATRIX_Lattice_Coupled):
    """TEAPOT implementation of coupled matrix lattice."""
    
    def __init__(self, teapot_lattice, bunch, name=None, parameterization='LB'):
        MATRIX_Lattice_Coupled.__init__(self, name, parameterization=parameterization)
        
        if not isinstance(teapot_lattice, TEAPOT_Lattice):
            orbitFinalize("`teapot_lattice` must be TEAPOT_Lattice instance.")
    
        if name is None:
            name = teapot_lattice.getName()
        self.setName(name)
        self.teapot_lattice = teapot_lattice
        self.bunch = Bunch()
        self.lost_bunch = Bunch()
        bunch.copyEmptyBunchTo(self.bunch)
        bunch.copyEmptyBunchTo(self.lost_bunch)
        self.matrixGenerator = MatrixGenerator()

        def twiss_action(params_dict):
            node = params_dict["node"]
            bunch = params_dict["bunch"]
            active_index = node.getActivePartIndex()
            n_parts = node.getnParts()
            length = node.getLength(active_index)
            if isinstance(node, BaseTEAPOT) and not isinstance(node, RingRFTEAPOT):
                self.matrixGenerator.initBunch(bunch)
                node.track(params_dict)
                matrix_node = BaseMATRIX(node.getName() + "_" + str(active_index))
                matrix_node.addParam("matrix_parent_node", node)
                matrix_node.addParam("matrix_parent_node_type", node.getType())
                matrix_node.addParam("matrix_parent_node_n_nodes", n_parts)
                matrix_node.addParam("matrix_parent_node_active_index", active_index)
                matrix_node.setLength(length)
                self.matrixGenerator.calculateMatrix(bunch, matrix_node.getMatrix())
                self.addNode(matrix_node)
            if isinstance(node, RingRFTEAPOT):
                rf_node = RingRFTEAPOT(node.getName())
                rf_node.setParamsDict(node.getParamsDict().copy())
                self.addNode(rf_node)

        actions = AccActionsContainer()
        actions.addAction(twiss_action, AccActionsContainer.BODY)
        params_dict = dict()
        params_dict["bunch"] = self.bunch
        params_dict["lostbunch"] = self.lost_bunch
        params_dict["position"] = 0.0
        params_dict["useCharge"] = self.teapot_lattice.getUseRealCharge()
        self.teapot_lattice.trackActions(actions, params_dict)
        self.makeOneTurnMatrix()
        self.initialize()

    def getKinEnergy(self):
        return self.bunch.getSyncParticle().kinEnergy()

    def rebuild(self, Ekin=-1.0):
        if Ekin > 0.0:
            self.bunch.getSyncParticle().kinEnergy(Ekin)
        for matrix_node in self.getNodes():
            if isinstance(matrix_node, BaseMATRIX) == True:
                node = matrix_node.getParam("matrix_parent_node")
                active_index = matrix_node.getParam("matrix_parent_node_active_index")
                n_parts = matrix_node.getParam("matrix_parent_node_n_nodes")
                if n_parts != node.getnParts():
                    orbitFinalize("`rebuild` stopped at node '{}'".format(node.getName()))
                self.matrixGenerator.initBunch(self.bunch)
                params_dict = dict()
                params_dict["bunch"] = self.bunch
                params_dict["node"] = node
                node.setActivePartIndex(active_index)
                node.track(params_dict)
                self.matrixGenerator.calculateMatrix(self.bunch, matrix_node.getMatrix())
        self.makeOneTurnMatrix()

    def getRingParametersDict(self, node=None):
        return MATRIX_Lattice_Coupled.getRingParametersDict(
            self, 
            momentum=self.bunch.getSyncParticle().momentum(), 
            mass=self.bunch.getSyncParticle().mass(), 
            node=node,
        )
    
    def getRingMatrix(self):
        return MATRIX_Lattice.getOneTurnMatrix(self)
    
    def getRingOrbit(self, z0):
        return self.trackOrbit(z0)

    def getRingTwissData(self):
        data = dict()
        data["s"] = []
        position = 0.0
        
        def add_data(params):
            for key, value in params.items():
                if key not in data:
                    data[key] = []
                data[key].append(value)
            data["s"].append(position)

        add_data(self.getRingParametersDict())
        for node in self.getNodes():
            if isinstance(node, BaseMATRIX):
                position += node.getLength()
                if node.getLength() > 0.0:                    
                    add_data(self.getRingParametersDict(node=node))
                    
        for key in [
            "momentum", 
            "mass", 
            "kin_energy",
            "period", 
            "frequency",
            "M", 
            "V",
            "eigvals", 
            "eigvecs", 
            "eigtunes", 
            "stable", 
            "coupled", 
        ]:
            data.pop(key)
        return data
            
    def getRingDispersionData(self):
        raise NotImplementedError

    def getChromaticities(self):
        raise NotImplementedError