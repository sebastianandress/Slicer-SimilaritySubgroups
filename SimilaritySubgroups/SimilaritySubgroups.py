import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import numpy.matlib as npm
import math
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans,vq,whiten

#
# SimilaritySubgroups
#

class SimilaritySubgroups(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Similarity Subgroups" 
    self.parent.categories = ["Registration"]
    self.parent.dependencies = []
    self.parent.contributors = ["Sebastian Andress (LMU Munich)"]
    self.parent.helpText = ""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = ""

#
# SimilaritySubgroupsWidget
#

class SimilaritySubgroupsWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    self.logic = SimilaritySubgroupsLogic()
    self.logic.logCallback = self.addLog

    # Instantiate and connect widgets ...
    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    self.preRegistrationCB = qt.QCheckBox("Use Pre-Registration")
    self.preRegistrationCB.checked = DEFAULT_PREREGISTRATION
    parametersFormLayout.addRow(self.preRegistrationCB)

    # landmark selectors
    self.sourceLandmarkSelector = slicer.qMRMLNodeComboBox()
    self.sourceLandmarkSelector.enabled = DEFAULT_PREREGISTRATION
    self.sourceLandmarkSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.sourceLandmarkSelector.selectNodeUponCreation = True
    self.sourceLandmarkSelector.addEnabled = False
    self.sourceLandmarkSelector.removeEnabled = False
    self.sourceLandmarkSelector.noneEnabled = False
    self.sourceLandmarkSelector.showHidden = False
    self.sourceLandmarkSelector.showChildNodeTypes = False
    self.sourceLandmarkSelector.setMRMLScene( slicer.mrmlScene )
    self.sourceLandmarkSelector.setToolTip( "Pick the source landmars to the algorithm." )
    parametersFormLayout.addRow("  Source Landmarks: ", self.sourceLandmarkSelector)

    self.targetLandmarkSelector = slicer.qMRMLNodeComboBox()
    self.targetLandmarkSelector.enabled = DEFAULT_PREREGISTRATION
    self.targetLandmarkSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
    self.targetLandmarkSelector.selectNodeUponCreation = True
    self.targetLandmarkSelector.addEnabled = False
    self.targetLandmarkSelector.removeEnabled = False
    self.targetLandmarkSelector.noneEnabled = False
    self.targetLandmarkSelector.showHidden = False
    self.targetLandmarkSelector.showChildNodeTypes = False
    self.targetLandmarkSelector.setMRMLScene( slicer.mrmlScene )
    self.targetLandmarkSelector.setToolTip( "Pick the target landmars to the algorithm." )
    parametersFormLayout.addRow("  Target Landmarks: ", self.targetLandmarkSelector)

    # source model selector
    self.sourceModelSelector = slicer.qMRMLNodeComboBox()
    self.sourceModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.sourceModelSelector.selectNodeUponCreation = True
    self.sourceModelSelector.addEnabled = False
    self.sourceModelSelector.removeEnabled = False
    self.sourceModelSelector.noneEnabled = False
    self.sourceModelSelector.showHidden = False
    self.sourceModelSelector.showChildNodeTypes = False
    self.sourceModelSelector.setMRMLScene( slicer.mrmlScene )
    self.sourceModelSelector.setToolTip( "Pick the source model to the algorithm." )
    parametersFormLayout.addRow("Source Model: ", self.sourceModelSelector)

    # target model selector
    self.targetModelSelector = slicer.qMRMLNodeComboBox()
    self.targetModelSelector.nodeTypes = ["vtkMRMLModelNode"]
    self.targetModelSelector.selectNodeUponCreation = True
    self.targetModelSelector.addEnabled = False
    self.targetModelSelector.removeEnabled = False
    self.targetModelSelector.noneEnabled = False
    self.targetModelSelector.showHidden = False
    self.targetModelSelector.showChildNodeTypes = False
    self.targetModelSelector.setMRMLScene( slicer.mrmlScene )
    self.targetModelSelector.setToolTip( "Pick the target model to the algorithm." )
    parametersFormLayout.addRow("Target Model: ", self.targetModelSelector)
    
    self.initializationClusterRadiusSelector = ctk.ctkSliderWidget()
    self.initializationClusterRadiusSelector.singleStep = 0.1
    self.initializationClusterRadiusSelector.minimum = 0
    self.initializationClusterRadiusSelector.maximum = 1000 # TODO as expansion of Model
    self.initializationClusterRadiusSelector.value = DEFAULT_INITIALIZATIONCLUSTERRADIUS
    self.initializationClusterRadiusSelector.suffix = 'mm'
    parametersFormLayout.addRow("Initialization Cluster Radius: ", self.initializationClusterRadiusSelector)

    self.initializationIterationsSelector = ctk.ctkSliderWidget()
    self.initializationIterationsSelector.singleStep = 1
    self.initializationIterationsSelector.minimum = 0
    self.initializationIterationsSelector.maximum = 100
    self.initializationIterationsSelector.value = DEFAULT_INITIALIZATIONITERATIONS
    parametersFormLayout.addRow("Initialization Iterations: ", self.initializationIterationsSelector)

    self.minimalClusterAreaSelector = ctk.ctkSliderWidget()
    self.minimalClusterAreaSelector.singleStep = 0.1
    self.minimalClusterAreaSelector.minimum = 1
    self.minimalClusterAreaSelector.maximum = 50000
    self.minimalClusterAreaSelector.value = DEFAULT_MINIMALCLUSTERAREA
    self.minimalClusterAreaSelector.suffix = 'mm2'
    parametersFormLayout.addRow("Min. Cluster Area: ", self.minimalClusterAreaSelector)

    self.openingWidthSelector = ctk.ctkSliderWidget()
    self.openingWidthSelector.singleStep = 1
    self.openingWidthSelector.minimum = 1
    self.openingWidthSelector.maximum = 30
    self.openingWidthSelector.value = DEFAULT_OPENINGWIDTH
    parametersFormLayout.addRow("Opening Width: ", self.openingWidthSelector)

    self.cutoffThresholdSelector = ctk.ctkSliderWidget()
    self.cutoffThresholdSelector.singleStep = 0.01
    self.cutoffThresholdSelector.minimum = 0.01
    self.cutoffThresholdSelector.maximum = 10
    self.cutoffThresholdSelector.value = DEFAULT_CUTOFFTHRESHOLD
    self.cutoffThresholdSelector.suffix = 'mm'
    parametersFormLayout.addRow("Cutoff Threshold: ", self.cutoffThresholdSelector)

    self.maximalIterationsSelector = ctk.ctkSliderWidget()
    self.maximalIterationsSelector.singleStep = 1
    self.maximalIterationsSelector.minimum = 1
    self.maximalIterationsSelector.maximum = 100
    self.maximalIterationsSelector.value = DEFAULT_MAXIMALITERATIONS
    parametersFormLayout.addRow("Max. Iterations: ", self.maximalIterationsSelector)

    self.addClustersCB = qt.QCheckBox("Mark Clusters")
    self.addClustersCB.checked = DEFAULT_ADDCLUSTERS
    self.deleteScalarsCB = qt.QCheckBox("Delete old Scalars")
    self.deleteScalarsCB.checked = DEFAULT_DELETESCALARS
    parametersFormLayout.addRow(self.addClustersCB)
    parametersFormLayout.addRow(self.deleteScalarsCB)

    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)


    ## connections ##

    # Parameters
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.preRegistrationCB.connect('toggled(bool)', self.updateLayout)
    self.sourceModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateLayout)
    self.targetModelSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateLayout)
    self.sourceLandmarkSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateLayout)
    self.targetLandmarkSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateLayout)

    ## Add vertical spacer
    self.layout.addStretch(1)
    self.updateLayout()

  def cleanup(self):
    pass


  def onApplyButton(self):

    if self.applyButton.text == 'Cancel':
      self.logic.requestCancel()
      self.addLog('Cancel requested...')
      return

    self.applyButton.text = 'Cancel'
    errorMessage = None
    qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    self.logic.sourceModel = self.sourceModelSelector.currentNode() 
    self.logic.targetModel = self.targetModelSelector.currentNode()
    self.logic.sourceLM = self.sourceLandmarkSelector.currentNode() if self.preRegistrationCB.checked else None
    self.logic.targetLM = self.targetLandmarkSelector.currentNode() if self.preRegistrationCB.checked else None
    self.logic.initializationClusterRadius = self.initializationClusterRadiusSelector.value
    self.logic.initializationIterations = int(self.initializationIterationsSelector.value)
    self.logic.minimalClusterArea = self.minimalClusterAreaSelector.value
    self.logic.openingWidth = int(self.openingWidthSelector.value)
    self.logic.cutoffThreshold = self.cutoffThresholdSelector.value
    self.logic.maximalIterations = int(self.maximalIterationsSelector.value)
    self.logic.addClusters = self.addClustersCB.checked
    self.logic.deleteScalars = self.deleteScalarsCB.checked
    try:
      self.logic.run()
    except Exception as e:
      import traceback
      traceback.print_exc()
      errorMessage = str(e)

    slicer.util.showStatusMessage("")
    self.applyButton.text = 'Apply'
    qt.QApplication.restoreOverrideCursor()
    if errorMessage:
      slicer.util.errorDisplay("Registration failed: " + errorMessage)

  
  def updateLayout(self):
    self.sourceLandmarkSelector.enabled = self.preRegistrationCB.checked
    self.targetLandmarkSelector.enabled = self.preRegistrationCB.checked
    
  def addLog(self, text):
    slicer.util.showStatusMessage(text)
    slicer.app.processEvents() # force update



#
# SimilaritySubgroupsLogic
#

class SimilaritySubgroupsLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    # inputs
    self.sourceModel = None
    self.targetModel = None
    self.sourceLM = None
    self.targetLM = None
    self.initializationClusterRadius = DEFAULT_INITIALIZATIONCLUSTERRADIUS
    self.initializationIterations = DEFAULT_INITIALIZATIONITERATIONS
    self.minimalClusterArea = DEFAULT_MINIMALCLUSTERAREA
    self.openingWidth = DEFAULT_OPENINGWIDTH
    self.cutoffThreshold = DEFAULT_CUTOFFTHRESHOLD
    self.maximalIterations = DEFAULT_MAXIMALITERATIONS
    self.addClusters = DEFAULT_ADDCLUSTERS
    self.deleteScalars = DEFAULT_DELETESCALARS

    self.logCallback = None
    self.cancelRequested = False

  def requestCancel(self):
    logging.info("User requested cancelling.")
    self.cancelRequested = True

  def _log(self, message):
    if self.logCallback:
      self.logCallback(message)

  def run(self):
    self.cancelRequested = False
    ar_names = list()
    for a in range(self.sourceModel.GetPolyData().GetPointData().GetNumberOfArrays()):
      ar_names.append(self.sourceModel.GetPolyData().GetPointData().GetArrayName(a))
    for a in reversed(ar_names):
      if self.deleteScalars:
        self.sourceModel.GetPolyData().GetPointData().RemoveArray(a)
      elif a == None:
        continue
      elif LET_CLUSTER in a or a in [LET_INTERMEDIATEDISTANCES, LET_INTERMEDIATEIDS, LET_INTERMEDIATEREGISTERED]:
        self.sourceModel.GetPolyData().GetPointData().RemoveArray(a)

    # initial landmark registration
    if self.sourceLM and self.targetLM:
      lmTrf = self._regLandmark(self.sourceLM, self.targetLM)
    else:
      lmTrf = vtk.vtkTransform()

    sourcePD = vtk.vtkPolyData()
    sourcePD.DeepCopy(self._transformPD(self.sourceModel.GetPolyData(), lmTrf))
    # sourcePD.DeepCopy(self.sourceModel.GetPolyData())

    idArrayId = self._addArray(sourcePD, range(sourcePD.GetNumberOfPoints()), LET_INTERMEDIATEIDS)
    regArrayId = self._addArray(sourcePD, [0]*(sourcePD.GetNumberOfPoints()), LET_INTERMEDIATEREGISTERED)


    clusters = 1
    for it in range(self.maximalIterations):
      self._log("Iteration %s/%s: %s clusters" % (it, self.maximalIterations, clusters-1))
      if self.cancelRequested:
        break
      smallestTrf = vtk.vtkTransform()
      smallestDev = -1
      smallestIds = list()

      # extract remain model
      remainModel = self._thresholdPD(sourcePD, LET_INTERMEDIATEREGISTERED, 0,0)

      # check for large enough clusters
      remPDList = self._connectivityPD(remainModel, largest=False)
      remPDSizes = list()
      for remPD in remPDList:
        remPDSizes.append(self._getSurfaceArea(remPD))
      
      for i, size in enumerate(remPDSizes):
        if size < self.minimalClusterArea:
          continue

        for _ in range(self.initializationIterations):
          # extract random piece
          vertexId = np.random.randint(0,remPDList[i].GetNumberOfPoints())
          extractPD = self._extractDistancePD(remPDList[i], vertexId, self.initializationClusterRadius)

          # register piece
          trf = self._regICP(extractPD, self.targetModel.GetPolyData())
          extractTrfPD = self._transformPD(extractPD, trf)

          # get/save mean deviation/piece/trf
          dev = self._calculateDistances(extractTrfPD, self.targetModel.GetPolyData())
          meanDev = np.mean(dev)

          if smallestDev == -1 or smallestDev > meanDev:
            smallestDev = meanDev
            smallestTrf = trf
            smallestIds = vtk.util.numpy_support.vtk_to_numpy(extractTrfPD.GetPointData().GetArray(self.getArrayNumber(extractPD, LET_INTERMEDIATEIDS)))

      # breaks loop if no piece remaining or all pieces smaller then self.minimalClusterArea
      print('smallest dev: ', smallestDev)
      if smallestDev == -1:
        break
      
      # register whole model, calculate deviations
      trfPD = self._transformPD(sourcePD, smallestTrf)
      dev = self._calculateDistances(trfPD, self.targetModel.GetPolyData())
      self._addArray(trfPD, dev, LET_INTERMEDIATEDISTANCES)

      # extract all pieces with distance <x
      extractPD = self._thresholdPD(trfPD, LET_INTERMEDIATEDISTANCES, 0, self.cutoffThreshold)
      extractPD.GetPointData().RemoveArray(LET_INTERMEDIATEDISTANCES)

      for sub in range(self.openingWidth+1):
        sgPD = self._openingPD(extractPD, self.openingWidth-sub)
        extractPdList = self._connectivityPD(sgPD, largest=False)
        try:
          mcId = self._regionMostCommonIds(extractPdList, LET_INTERMEDIATEIDS, smallestIds, self.minimalClusterArea)
        except IndexError:
          continue

        sgPD = extractPdList[mcId]
        print('shrinkgrow: ', self.openingWidth-sub, self._getSurfaceArea(sgPD))
        
        if self._getSurfaceArea(sgPD) >= self.minimalClusterArea:
          break
          
      if self._getSurfaceArea(sgPD) < self.minimalClusterArea:
        continue
      

      trf = self._regICP(sgPD, self.targetModel.GetPolyData())
      trfPD = self._transformPD(trfPD, trf)
      
      dev = self._calculateDistances(trfPD, self.targetModel.GetPolyData())
      self._addArray(self.sourceModel.GetPolyData(), dev, "deviation" + LET_CLUSTER + str(clusters))
      
      # mark island as registered
      eia = vtk.util.numpy_support.vtk_to_numpy(sgPD.GetPointData().GetArray(self.getArrayNumber(sgPD, LET_INTERMEDIATEIDS)))
      sra = vtk.util.numpy_support.vtk_to_numpy(sourcePD.GetPointData().GetArray(LET_INTERMEDIATEREGISTERED))
      
      np.put(sra, eia, 1)
      regArrayId = self._addArray(sourcePD, sra, LET_INTERMEDIATEREGISTERED)
      

      if self.addClusters:
        # self._addModel(sgPD, self.sourceModel.GetName() + "_cluster_"+str(clusters))
        ca = np.zeros(sourcePD.GetNumberOfPoints())
        np.put(ca, eia, 1)
        self._addArray(self.sourceModel.GetPolyData(), ca, "mark" + LET_CLUSTER + str(clusters))
      clusters += 1



  def cleanup(self):
    pass
  
  #
  # Deform Models/Fiducials for Experiments
  #
  def createIslandDeformation(self, model, fiducials, idx, nr, height):
    # calculate normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(model.GetPolyData())
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    normalsArray = vtk.util.numpy_support.vtk_to_numpy(normals.GetOutput().GetPointData().GetArray('Normals'))
    
    deformedPD = vtk.vtkPolyData()
    deformedPD.DeepCopy(model.GetPolyData())

    expandIds = self._expandIds(normals.GetOutput(), idx, nr)

    deformedArray = np.zeros(deformedPD.GetNumberOfPoints())
    
    # move along their normals
    for i in expandIds:
      pos = np.array(deformedPD.GetPoints().GetPoint(i))
      newPos = pos + normalsArray[i]*height
      deformedPD.GetPoints().SetPoint(i,newPos.tolist())
      deformedArray[i] = 1
    deformation = self._calculateDistances(deformedPD, model.GetPolyData())
    deformedPD.GetPointData().GetArray(deformedPD.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(deformation))).SetName("Deformation")
    deformedPD.GetPointData().GetArray(deformedPD.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(deformedArray))).SetName("Deformed")

    fiducialVertexIds = self._getClosestVertexIds(model, fiducials)
    fid = slicer.vtkMRMLMarkupsFiducialNode()
    for i, ii in enumerate(fiducialVertexIds):
      pos = [0,0,0]
      fiducials.GetNthFiducialPosition(i, pos)
      if ii in expandIds:
        pos = np.array(pos) + normalsArray[ii]*height
      n = fid.AddFiducialFromArray(pos)
      fid.SetNthFiducialLabel(n, fiducials.GetNthFiducialLabel(i))

    # add all to scene
    mod = slicer.modules.models.logic().AddModel(deformedPD)
    mod.SetName(model.GetName() + '_island')
    fid.SetName(fiducials.GetName() + '_island')
    slicer.mrmlScene.AddNode(fid)

    return mod, fid

  def createSpikesDeformation(self, model, fiducials, nr, height):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(model.GetPolyData())
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    normalsArray = vtk.util.numpy_support.vtk_to_numpy(normals.GetOutput().GetPointData().GetArray('Normals'))
    
    deformedPD = vtk.vtkPolyData()
    deformedPD.DeepCopy(model.GetPolyData())

    expandIds = np.random.default_rng().choice(len(normalsArray), size=nr, replace=False)

    fiducialVertexIds = self._getClosestVertexIds(model, fiducials)

    deformationArray = np.zeros(deformedPD.GetNumberOfPoints())
    
    # move along their normals
    for i in expandIds:
      pos = np.array(deformedPD.GetPoints().GetPoint(i))
      newPos = pos + normalsArray[i]*height
      deformedPD.GetPoints().SetPoint(i,newPos.tolist())
      deformationArray[i] = height
    ar = deformedPD.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(deformationArray))
    deformedPD.GetPointData().GetArray(ar).SetName("Deformation")

    fid = slicer.vtkMRMLMarkupsFiducialNode()
    for i, ii in enumerate(fiducialVertexIds):
      pos = [0,0,0]
      fiducials.GetNthFiducialPosition(i, pos)
      if ii in expandIds:
        pos = pos + normalsArray[ii]*height
      n = fid.AddFiducialFromArray(pos)
      fid.SetNthFiducialLabel(n, fiducials.GetNthFiducialLabel(i))

    # add all to scene
    mod = slicer.modules.models.logic().AddModel(deformedPD)
    mod.SetName(model.GetName() + '_spikes')
    fid.SetName(fiducials.GetName() + '_spikes')
    slicer.mrmlScene.AddNode(fid)

    return mod, fid

  def createShearDeformation(self, model, fiducials, shearAngle, shearAxis):
    deformedPD = vtk.vtkPolyData()
    deformedPD.DeepCopy(model.GetPolyData())

    pointTree = vtk.vtkPointLocator()
    pointTree.SetDataSet(model.GetPolyData())
    pointTree.BuildLocator()

    ## Shear ##
    trfCenter = vtk.vtkTransform()
    trfCenter.Translate(deformedPD.GetCenter())

    trfCenterInv = vtk.vtkTransform()
    trfCenterInv.Translate(deformedPD.GetCenter())
    trfCenterInv.Inverse()

    shearMat = vtk.vtkMatrix4x4() # https://www.wisdomjobs.com/userfiles/3(15).png
    shear = math.tan(math.radians(shearAngle))
    if 'y' in shearAxis:
      shearMat.SetElement(0,2,shear) # sh(zx)
    if 'z' in shearAxis:
      shearMat.SetElement(1,0,shear) # sh(xy)
    if 'x' in shearAxis:
      shearMat.SetElement(2,1,shear) # sh(yz)

    trfShear = vtk.vtkTransform()
    trfShear.SetMatrix(shearMat)

    trf = self._multiplyTransforms([trfCenterInv, trfShear, trfCenter])

    deformedPD = self._transformPD(deformedPD, trf)
    deformedFidPoints = vtk.vtkPoints()
    slicer.modules.markupstomodel.logic().MarkupsToPoints(fiducials, deformedFidPoints)
    deformedFidPD = vtk.vtkPolyData()
    deformedFidPD.SetPoints(deformedFidPoints)
    deformedFidPD = self._transformPD(deformedFidPD, trf)

    fid = slicer.vtkMRMLMarkupsFiducialNode()
    for p in range(deformedFidPD.GetNumberOfPoints()):
      n = fid.AddFiducialFromArray(deformedFidPD.GetPoint(p))
      fid.SetNthFiducialLabel(n, fiducials.GetNthFiducialLabel(p))

    mod = slicer.modules.models.logic().AddModel(deformedPD)
    slicer.mrmlScene.AddNode(fid)
    mod.SetName(model.GetName() + '_shear')
    fid.SetName(fiducials.GetName() + '_shear')

    return mod, fid

  def createShiftDeformation(self, model, fiducials, idx, nr, height, rotation):

    deformedPD = vtk.vtkPolyData()
    deformedPD.DeepCopy(model.GetPolyData())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(model.GetPolyData())
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()

    # expand ids
    extractIds = self._expandIds(deformedPD, idx, nr)
    # extract ids
    extractPD = self._extractPD(deformedPD, extractIds)
    
    # get center of extraction
    center = extractPD.GetCenter()
    # trf extraction to origin
    trfFromOrigin = vtk.vtkTransform()
    trfFromOrigin.Translate(center)

    trfToOrigin = vtk.vtkTransform()
    trfToOrigin.Translate(center)
    trfToOrigin.Inverse()

    trf = vtk.vtkTransform()
    trf.RotateX(rotation[0])
    trf.RotateY(rotation[1])
    trf.RotateZ(rotation[2])
    transl = np.array(normals.GetOutput().GetPointData().GetNormals().GetTuple(idx)) * height
    trf.Translate(transl)

    trfFinal = self._multiplyTransforms([trfToOrigin, trf, trfFromOrigin])

    deformedArray = np.zeros(deformedPD.GetNumberOfPoints())
    for i in extractIds:
      pos = np.array(deformedPD.GetPoints().GetPoint(i))
      trfPoint = trfFinal.TransformPoint(pos)
      deformedPD.GetPoints().SetPoint(i,trfPoint)
      deformedArray[i] = 1
    deformation = self._calculateDistances(deformedPD, model.GetPolyData())
    deformedPD.GetPointData().GetArray(deformedPD.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(deformation))).SetName("Deformation")
    deformedPD.GetPointData().GetArray(deformedPD.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(deformedArray))).SetName("Deformed")

    fiducialVertexIds = self._getClosestVertexIds(model, fiducials)
    fid = slicer.vtkMRMLMarkupsFiducialNode()
    for i, ii in enumerate(fiducialVertexIds):
      pos = [0,0,0]
      fiducials.GetNthFiducialPosition(i, pos)
      if ii in extractIds:
        pos = np.array(pos) + (-np.array(model.GetPolyData().GetPoint(ii))+np.array(deformedPD.GetPoint(ii)))
      n = fid.AddFiducialFromArray(pos)
      fid.SetNthFiducialLabel(n, fiducials.GetNthFiducialLabel(i))

    # add all to scene
    mod = slicer.modules.models.logic().AddModel(deformedPD)
    mod.SetName(model.GetName() + '_shift')
    fid.SetName(fiducials.GetName() + '_shift')
    slicer.mrmlScene.AddNode(fid)

    return mod, fid

  #
  # Method's functions
  #
  @staticmethod
  def _openingPD (polydata, width):
    baseArId = polydata.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(range(polydata.GetNumberOfPoints())))
    basePd = vtk.vtkPolyData()
    basePd.DeepCopy(polydata)

    pd = vtk.vtkPolyData()
    pd.DeepCopy(polydata)

    # erode
    erodewidth = 0
    for w in range(width):

      arId = pd.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(range(pd.GetNumberOfPoints())))

      featureEdges = vtk.vtkFeatureEdges()
      featureEdges.SetInputData(pd)
      featureEdges.BoundaryEdgesOn()
      featureEdges.FeatureEdgesOff()
      featureEdges.ManifoldEdgesOff()
      featureEdges.NonManifoldEdgesOff()
      featureEdges.Update()

      ar = vtk.util.numpy_support.vtk_to_numpy(featureEdges.GetOutput().GetPointData().GetArray(arId))
      pd.GetPointData().RemoveArray(arId)
      ids = vtk.vtkIdTypeArray()
      ids.SetNumberOfComponents(1)
      for i in ar:
        ids.InsertNextValue(i)


      selectionNode = vtk.vtkSelectionNode()
      selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
      selectionNode.SetContentType(4)
      selectionNode.SetSelectionList(ids)
      selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)
      selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(),1)

      selection = vtk.vtkSelection()
      selection.AddNode(selectionNode)

      extractSelection = vtk.vtkExtractSelection()
      extractSelection.SetInputData(0, pd)
      extractSelection.SetInputData(1, selection)
      extractSelection.Update()

      geometryFilter = vtk.vtkGeometryFilter() 
      geometryFilter.SetInputData(extractSelection.GetOutput()) 
      geometryFilter.Update()

      cleanFilter = vtk.vtkCleanPolyData()
      cleanFilter.SetInputData(geometryFilter.GetOutput())
      cleanFilter.Update()

      if cleanFilter.GetOutput().GetNumberOfPoints() == 0:
        break
      
      erodewidth = w + 1
      pd.DeepCopy(cleanFilter.GetOutput())

    # dilate
    baseAr = set(vtk.util.numpy_support.vtk_to_numpy(pd.GetPointData().GetArray(baseArId)).tolist())
    for _ in range(erodewidth):

      featureEdges = vtk.vtkFeatureEdges()
      featureEdges.SetInputData(pd)
      featureEdges.BoundaryEdgesOn()
      featureEdges.FeatureEdgesOff()
      featureEdges.ManifoldEdgesOff()
      featureEdges.NonManifoldEdgesOff()
      featureEdges.Update()

      ar = vtk.util.numpy_support.vtk_to_numpy(featureEdges.GetOutput().GetPointData().GetArray(baseArId))
      for i in ar:
        conCellsIdList = vtk.vtkIdList()
        basePd.GetPointCells(i, conCellsIdList)

        for c in range(conCellsIdList.GetNumberOfIds()):
          cellsPointIds = basePd.GetCell(conCellsIdList.GetId(c)).GetPointIds()
        
          for p in range(cellsPointIds.GetNumberOfIds()):
            baseAr.add(cellsPointIds.GetId(p))
      
      ids = vtk.vtkIdTypeArray()
      ids.SetNumberOfComponents(1)
      for i in baseAr:
          ids.InsertNextValue(i)
      selectionNode = vtk.vtkSelectionNode()
      selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
      selectionNode.SetContentType(4)
      selectionNode.SetSelectionList(ids)
      selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

      selection = vtk.vtkSelection()
      selection.AddNode(selectionNode)

      extractSelection = vtk.vtkExtractSelection()
      extractSelection.SetInputData(0, basePd)
      extractSelection.SetInputData(1, selection)
      extractSelection.Update()

      geometryFilter = vtk.vtkGeometryFilter() 
      geometryFilter.SetInputData(extractSelection.GetOutput()) 
      geometryFilter.Update()

      cleanFilter = vtk.vtkCleanPolyData()
      cleanFilter.SetInputData(geometryFilter.GetOutput())
      cleanFilter.Update()

      pd.DeepCopy(cleanFilter.GetOutput())

    pd.GetPointData().RemoveArray(baseArId)
    return pd

  @staticmethod
  def _expandIds(polydata, vertexID, nr):
    if polydata.GetNumberOfPoints() <= nr:
      return list(range(polydata.GetNumberOfPoints()))

    expansionIds = [vertexID]
    checkedIds = []
    noneAdded = False
    while noneAdded == False: #len(expansionIds) < nr and len(expansionIds) != len(checkedIds):
      noneAdded = True
      for curId in expansionIds:
        if len(expansionIds) >= nr:
          break
        if curId in checkedIds:
          continue
        else:
          checkedIds.append(curId)

        conCellsIdList = vtk.vtkIdList()
        polydata.GetPointCells(curId, conCellsIdList)

        for c in range(conCellsIdList.GetNumberOfIds()):
          cellsPointIds = polydata.GetCell(conCellsIdList.GetId(c)).GetPointIds()
          for p in range(cellsPointIds.GetNumberOfIds()):

            if len(expansionIds) >= nr:
              break
            
            if cellsPointIds.GetId(p) in expansionIds:
              continue

            else:
              expansionIds.append(cellsPointIds.GetId(p))
              noneAdded = False

    return expansionIds
  

  def _extractDistancePD(self, polydata, seedId, distance):
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1)
    sphere.SetCenter(polydata.GetPoint(seedId))
    sphere.Update()
    
    if polydata.GetPointData().GetArray(LET_INTERMEDIATEIDS):
      seedId = polydata.GetPointData().GetArray(LET_INTERMEDIATEIDS).GetValue(seedId)
    else:
      self._addArray(polydata, range(polydata.GetNumberOfPoints()), LET_INTERMEDIATEIDS)

    dist = vtk.vtkDistancePolyDataFilter()
    dist.AddInputData(0, polydata)
    dist.AddInputData(1, sphere.GetOutput())
    dist.SignedDistanceOff()
    dist.NegateDistanceOff()
    dist.ComputeSecondDistanceOff()
    dist.Update()
    extractPd = self._thresholdPD(dist.GetOutput(), "Distance", upper=distance)
    
    extrSeedId = vtk.util.numpy_support.vtk_to_numpy(extractPd.GetPointData().GetArray(LET_INTERMEDIATEIDS)).tolist().index(seedId)
    
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(extractPd)
    connectivityFilter.SetExtractionModeToPointSeededRegions()
    connectivityFilter.AddSeed(extrSeedId)
    connectivityFilter.Update()
    
    return connectivityFilter.GetOutput()

  @staticmethod
  def _thresholdPD(polydata, array, lower=0, upper=0):
    thresh = vtk.vtkThreshold()
    thresh.SetInputArrayToProcess(0, 0, vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes().SCALARS, array)
    thresh.SetInputData(polydata)
    thresh.ThresholdBetween(lower, upper)

    geom = vtk.vtkGeometryFilter()
    geom.SetInputConnection(thresh.GetOutputPort())

    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(geom.GetOutputPort())
    cleanFilter.Update()

    return cleanFilter.GetOutput()

  @staticmethod
  def _extractPD(polydata, pointIds, inverse=False):

    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfComponents(1)
    for i in pointIds:
      ids.InsertNextValue(i)

    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.POINT)
    selectionNode.SetContentType(4)
    selectionNode.SetSelectionList(ids)
    selectionNode.GetProperties().Set(vtk.vtkSelectionNode.CONTAINING_CELLS(), 1)

    if inverse == True:
      selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(),1)

    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, polydata)
    extractSelection.SetInputData(1, selection)
    # extractSelection.Update()

    geometryFilter = vtk.vtkGeometryFilter() 
    geometryFilter.SetInputConnection(extractSelection.GetOutputPort()) 
    # geometryFilter.Update()

    cleanFilter = vtk.vtkCleanPolyData()
    cleanFilter.SetInputConnection(geometryFilter.GetOutputPort())
    cleanFilter.Update()

    return cleanFilter.GetOutput()

  @staticmethod
  def _connectivityPD(polydata, largest=True):
    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(polydata)
    
    if largest:
      connectivityFilter.SetExtractionModeToLargestRegion()
      connectivityFilter.Update()

      cleanFilter = vtk.vtkCleanPolyData()
      cleanFilter.SetInputConnection(connectivityFilter.GetOutputPort())
      cleanFilter.Update()

      return cleanFilter.GetOutput()
    
    else:
      connectivityFilter.SetExtractionModeToAllRegions()
      connectivityFilter.ColorRegionsOn()
      connectivityFilter.Update()

      selectorCon = vtk.vtkThreshold()
      selectorCon.SetInputArrayToProcess(0, 0, vtk.vtkDataObject().FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes().SCALARS, "RegionId")
      selectorCon.SetInputConnection(connectivityFilter.GetOutputPort())
      selectorCon.AllScalarsOff()

      regions = list()

      for r in range(connectivityFilter.GetNumberOfExtractedRegions()):

        selectorCon.ThresholdBetween(r,r)
        geometryCon = vtk.vtkGeometryFilter()
        geometryCon.SetInputConnection(selectorCon.GetOutputPort())
        geometryCon.Update()
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputConnection(geometryCon.GetOutputPort())
        cleanFilter.Update()

        pd = vtk.vtkPolyData()
        pd.DeepCopy(cleanFilter.GetOutput())
        regions.append(pd)
      
      return regions

  @staticmethod
  def _transformPD(polydata, trf):
    transformPD = vtk.vtkTransformPolyDataFilter()
    transformPD.SetTransform(trf)
    transformPD.SetInputData(polydata)
    transformPD.Update()

    return transformPD.GetOutput()

  @staticmethod
  def _regLandmark(sourceLM, targetLM):

    def fiducialsToPoints(fiducials):
      points = vtk.vtkPoints()
      for fid in range(fiducials.GetNumberOfFiducials()):
        pos = [0]*3
        fiducials.GetNthFiducialPosition(fid, pos)
        points.InsertNextPoint(list(pos))
      return points
    
    targetPoints = fiducialsToPoints(targetLM)
    sourcePoints = fiducialsToPoints(sourceLM)

    landmarkTransform = vtk.vtkLandmarkTransform()
    landmarkTransform.SetSourceLandmarks(sourcePoints)
    landmarkTransform.SetTargetLandmarks(targetPoints)
    landmarkTransform.SetModeToRigidBody()
    landmarkTransform.Modified()
    landmarkTransform.Update()

    return landmarkTransform

  @staticmethod
  def _regICP(sourcePD, targetPD, pointsNr=0):

    icpTransform = vtk.vtkIterativeClosestPointTransform()
    icpTransform.SetSource(sourcePD)
    icpTransform.SetTarget(targetPD)
    icpTransform.GetLandmarkTransform().SetModeToRigidBody()
    icpTransform.StartByMatchingCentroidsOff()
    icpTransform.Modified()
    icpTransform.Update()

    return icpTransform

  @staticmethod
  def _calculateDistances(sourcePD, targetPD):
    dist = vtk.vtkDistancePolyDataFilter()
    dist.AddInputData(0, sourcePD)
    dist.AddInputData(1, targetPD)
    dist.SignedDistanceOff()
    dist.NegateDistanceOff()
    dist.ComputeSecondDistanceOff()
    dist.Update()
    
    return vtk.util.numpy_support.vtk_to_numpy(dist.GetOutput().GetPointData().GetArray("Distance"))

  @staticmethod
  def _regionMostCommonIds(regions, idArrayName, ids, minSize=0):
    for i in range(regions[0].GetPointData().GetNumberOfArrays()):
      if regions[0].GetPointData().GetArrayName(i) == idArrayName:
        break

    maxIntersects = -1
    r = -1
    for p, pd in enumerate(regions):
      ar = vtk.util.numpy_support.vtk_to_numpy(pd.GetPointData().GetArray(i))
  
      if len(np.intersect1d(ar,ids)) > 0 and len(ar) >= minSize:
        maxIntersects = len(np.intersect1d(ar,ids))
        r = p
    
    
    if r == -1:
      raise IndexError("No common region found.")
    
    return r

  @staticmethod
  def _getClosestVertexIds(model, fiducials):
    
    points = vtk.vtkPoints()
    slicer.modules.markupstomodel.logic().MarkupsToPoints(fiducials, points)
    
    tree = vtk.vtkPointLocator()
    tree.SetDataSet(model.GetPolyData())
    tree.BuildLocator()

    po = vtk.util.numpy_support.vtk_to_numpy(points.GetData())
    return [tree.FindClosestPoint(p) for p in po]

  @staticmethod
  def _getSurfaceArea(polydata):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(polydata)
    mass.Update() 

    return mass.GetSurfaceArea()


  # Helper
  @staticmethod
  def _addTrf(trf, name=None):
    no = slicer.vtkMRMLLinearTransformNode()
    no.SetAndObserveTransformFromParent(trf)
    t = slicer.mrmlScene.AddNode(no)
    if name:
      t.SetName(name)
    return t
  
  @staticmethod
  def _addModel(polydata, name=None):
    mod = slicer.modules.models.logic().AddModel(polydata)
    if name:
      mod.SetName(name)
    return mod
  
  @staticmethod
  def _addArray(polydata, values, name):
    arr = vtk.util.numpy_support.numpy_to_vtk(values)
    arr.SetName(name)
    polydata.GetPointData().RemoveArray(name)
    return polydata.GetPointData().AddArray(arr)

  @staticmethod
  def _fiducialsToList(fiducials):
    points = vtk.vtkPoints()
    slicer.modules.markupstomodel.logic().MarkupsToPoints(fiducials, points)
    return vtk.util.numpy_support.vtk_to_numpy(points.GetData())

  @staticmethod
  def _multiplyTransforms(trfList):
    def mult(trf1, trf2):
      dotMat = vtk.vtkMatrix4x4()
      vtk.vtkMatrix4x4().Multiply4x4(trf2.GetMatrix(), trf1.GetMatrix(), dotMat)
      trf = vtk.vtkTransform()
      trf.SetMatrix(dotMat)
      return trf
    
    outTrf = trfList[0]
    for t in range(len(trfList)-1):
      outTrf = mult(outTrf, trfList[t+1])
    
    return outTrf
  
  @staticmethod
  def getNode(name, node_type=None):
    try:
      node = slicer.util.getNode(name)
    except slicer.util.MRMLNodeNotFoundException:
      if node_type:
        if node_type == slicer.vtkMRMLModelNode:
          node = slicer.modules.models.logic().AddModel(vtk.vtkPolyData())
        else:
          n = node_type()
          node = slicer.mrmlScene.AddNode(n)
        node.SetName(name)
    return node

  @staticmethod
  def getArrayNumber(polydata, name):
    for i in range(polydata.GetPointData().GetNumberOfArrays()):
      if polydata.GetPointData().GetArrayName(i) == name:
        return i

    raise NameError('Array name not found in passed polydata')
  

  # old
  def _clusterTransforms(self, trfList, center, bandwidth=None):
    
    trfs = np.zeros((len(trfList), 6))
    for t, trf in enumerate(trfList):

      trfToCenter = vtk.vtkTransform()
      trfToCenter.Translate(*center)
      trfFromCenter = vtk.vtkTransform()
      trfFromCenter.Translate(*center)
      trfFromCenter.Inverse()

      trf_new = self._multiplyTransforms([trfToCenter, trf, trfFromCenter])

      pos = list(trf_new.GetPosition())
      ori = list(trf_new.GetOrientation())
      trfs[t] = np.array(pos+ori)

    clustering = MeanShift(bandwidth=bandwidth).fit(trfs)
    return clustering.labels_

  @staticmethod
  def _dbscanclusterTransforms(trfList, center, maxTransError=2, maxRotError=1):

    def similarity(x,y):
      tX = vtk.vtkTransform()
      tX.PostMultiply()
      tX.RotateWXYZ(*x[3:])
      tX.Translate(*x[:3])

      tY = vtk.vtkTransform()
      tY.PostMultiply()
      tY.RotateWXYZ(*y[3:])
      tY.Translate(*y[:3])

      pIn = center
      pXOut = [0,0,0]
      pYOut = [0,0,0]

      tX.TransformPoint(pIn, pXOut)
      tY.TransformPoint(pIn, pYOut)
      
      vIn = [1,0,0]
      vXOut = [0,0,0]
      vYOut = [0,0,0]
      tX.TransformVectorAtPoint(center, vIn, vXOut)
      tY.TransformVectorAtPoint(center, vIn, vYOut)
      
      rEr = np.degrees(np.arccos(np.clip(np.dot(vXOut, vYOut), -1.0, 1.0)))
      tEr = np.linalg.norm(np.array(pXOut) - np.array(pYOut))

      if tEr < maxTransError and rEr < maxRotError:
        er = ((tEr/maxTransError) + (rEr/maxRotError)) * 0.5
      else:
        er = (tEr/maxTransError) + (rEr/maxRotError)

      return er



    trfs = np.zeros((len(trfList), 7))
    for t, trf in enumerate(trfList):

      pos = list(trf.GetPosition())
      ori = list(trf.GetOrientationWXYZ())

      trfs[t] = np.array(pos+ori)

    clustering = DBSCAN(eps=1, min_samples=1, metric=similarity).fit(trfs)
    return clustering.labels_

class SimilaritySubgroupsTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    self.setUp()
    self.test_SimilaritySubgroups1()

  def test_SimilaritySubgroups1(self):

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    self.delayDisplay("Starting test_SimilaritySubgroups1")

    # from SegmentStatistics import SegmentStatisticsLogic

    self.delayDisplay("Create models and fiducials")

    sphere1 = vtk.vtkSphereSource()
    sphere1.SetRadius(30)
    sphere1.SetCenter(0,0,0)
    sphere1.SetPhiResolution(50)
    sphere1.SetThetaResolution(50)
    sphere1.Update()

    sphere2 = vtk.vtkSphereSource()
    sphere2.SetRadius(30)
    sphere2.SetCenter(10,0,0)
    sphere2.SetPhiResolution(50)
    sphere2.SetThetaResolution(50)
    sphere2.Update()

    sphere3 = vtk.vtkSphereSource()
    sphere3.SetRadius(30)
    sphere3.SetCenter(50,50,50)
    sphere3.SetPhiResolution(50)
    sphere3.SetThetaResolution(50)
    sphere3.Update()

    boolean = vtk.vtkBooleanOperationPolyDataFilter()
    boolean.SetOperationToUnion()
    boolean.SetInputData(0, sphere1.GetOutput())
    boolean.SetInputData(1, sphere2.GetOutput())
    boolean.Update()

    smod = slicer.modules.models.logic().AddModel(boolean.GetOutput())
    tmod = slicer.modules.models.logic().AddModel(sphere3.GetOutput())

    sfid = slicer.vtkMRMLMarkupsFiducialNode()
    slicer.mrmlScene.AddNode(sfid)
    sfid.AddFiducial(0,30,0)
    sfid.AddFiducial(0,-30,0)
    sfid.AddFiducial(-30,0,0)

    tfid = slicer.vtkMRMLMarkupsFiducialNode()
    slicer.mrmlScene.AddNode(tfid)
    tfid.AddFiducial(50,80,50)
    tfid.AddFiducial(50,20,50)
    tfid.AddFiducial(20,50,50)

    self.delayDisplay("Run algorithm")

    logic = SimilaritySubgroupsLogic()
    logic.sourceModel = smod
    logic.targetModel = tmod
    logic.sourceLM = sfid
    logic.targetLM = tfid
    logic.initializationClusterRadius = 20
    logic.initializationIterations = 5
    logic.minimalClusterArea = 10
    logic.openingWidth = 1
    logic.cutoffThreshold = 0.5
    logic.maximalIterations = 10
    logic.addClusters = True
    logic.deleteScalars = False
    logic.run()

    self.delayDisplay("Evaluate Result")

    arrays = list()
    for a in range(smod.GetPolyData().GetPointData().GetNumberOfArrays()):
      ar = smod.GetPolyData().GetPointData().GetArray(a)
      if ar.GetName().startswith('deviation_cluster'):
        arrays.append(vtk.util.numpy_support.vtk_to_numpy(ar))
    
    clusterDev = np.min(arrays, axis=0)

    meanMinDev = np.mean(clusterDev)
    self.assertLessEqual(meanMinDev, 0.5)

    self.delayDisplay('test_SimilaritySubgroups1 passed')



DEFAULT_INITIALIZATIONCLUSTERRADIUS = 50
DEFAULT_INITIALIZATIONITERATIONS = 3
DEFAULT_MINIMALCLUSTERAREA = 500
DEFAULT_OPENINGWIDTH = 8
DEFAULT_CUTOFFTHRESHOLD = 1.5
DEFAULT_MAXIMALITERATIONS = 10
DEFAULT_PREREGISTRATION = False
DEFAULT_ADDCLUSTERS = True
DEFAULT_DELETESCALARS = False

LET_CLUSTER = "_cluster_"
LET_INTERMEDIATEDISTANCES = "intermediate-distances"
LET_INTERMEDIATEREGISTERED = "intermediate-registered"
LET_INTERMEDIATEIDS = "intermediate-ids"

