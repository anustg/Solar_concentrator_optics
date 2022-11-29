# %%

# SCOAP
# Solar concentrator optics analyser in Python
# This code may contain errors and the author 
# v0.0(beta): written by Johannes Pottas, The Australian National University, 2022


import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import tkinter as tk
from tkinter import filedialog
import scipy as sp
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import statistics
import math


root = tk.Tk()
root.withdraw()

# Setting for minimum incircle/circumcirle ratio for triangles. Used to mask flat edge triangles.

# %%
minCircleRatio = .01

# Set up counter, relaxation factor for corrective rotations and starting points for while loop which performs reorientation of the data. No need to change anything unless the orientation loop fails to converge, then try reducing rotRelax (e.g. 0.5)
rotCount = 0
mux = 1
muy = 1
rotRelax = 1

## User file selection dialogue.

# %%
inPath = filedialog.askopenfilename()
inObject = open(inPath, 'r')

## User prompts for ideal focal length.

idFoc = int(input('Enter the ideal focal length of the panel [m]:'))*1000

## User prompts for X,Y,Z column indices.

xCol = int(input('Which column number contains X coordinates?'))
yCol = int(input('Which column number contains Y coordinates?'))
zCol = int(input('Which column number contains Z coordinates?'))

## User prompts for array dimensions. Currently hardcoded.

numCols = int(input('How many rows does the target pattern have?'))
numRows = int(input('How many columns does the target pattern have?'))

## Split lines into columns and build X,Y,Z arrays.

X, Y, Z = [], [], []
for dataline in open(inPath, 'r'):
    line = [float(s) for s in dataline.split()]
    X.append(line[xCol-1])
    Y.append(line[yCol-1])
    Z.append(line[zCol-1])

#Translate array so that mean coordinate coincides with the origin.

avX = (sum(X)/len(X))
X = [x - avX for x in X]
avY = (sum(Y)/len(Y))
Y = [y - avY for y in Y]
avZ = (sum(Z)/len(Z))
Z = [z - avZ for z in Z]

# A plot of the imported X,Y,Z data after translation. Comment in or out, as required.

inDataFig = plt.figure()
inDataAx = plt.axes(projection='3d')
inDataAx.scatter(X,Y,Z)
inDataAx.axis('tight')
inDataAx.set_xlabel('X [mm]')
inDataAx.set_ylabel('Y [mm]')
inDataAx.set_zlabel('Z [mm]')
inDataAx.set_title('Raw plot of '+inPath.rsplit('/', 1)[1])
plt.show()

## Storage for rotation adjustments.

muxAdjs = []
muyAdjs = []

while abs(mux) > 1e-4 or abs(muy) > 1e-4:

    rotCount = rotCount + 1

    XY = np.array([X,Y])
    XY = np.transpose(XY)

    ## 2D triangulation of X,Y points. Triangles with circle ratios of less than minCircleRatio are masked.  

    triXY = mtri.Triangulation(X, Y)
    CircleRatios= mtri.TriAnalyzer(triXY).circle_ratios(rescale=True)
    BorderMask = mtri.TriAnalyzer(triXY).get_flat_tri_mask(minCircleRatio)
    triXY.set_mask(BorderMask)

    ## .set_mask does not actually change the list of triangles. It only creates a boolean mask in the triangulation object. For later plotting in 3D,
    ## I create a new array without the flat triangles. There might be a more elegant way to do this.

    FlatTriInds = np.where(BorderMask)
    KeepTriangles = np.delete(triXY.triangles,FlatTriInds,axis=0)

    ## Calculate the measured surface normal at every point as the mean of the normals of the touching triangles. I need to check if another interpolator might be more appropriate.

    XYZ = np.array([X,Y,Z])
    XYZ = np.transpose(XYZ)
    
    # triXYZ = Delaunay(XYZ) # This 3D triangulation is not currently used, but may be useful for another purpose, so I'm leaving it here.

    intXYZ = mtri.CubicTriInterpolator(triXY, Z, trifinder=None) # This is the interpolator which may not be the best option for typical mirror panel curvature.
    (mx,my) = intXYZ.gradient(triXY.x,triXY.y)

    ## Compare mean of gradient from the last iteration to the new value and change direction if diverging.

    muxLast = mux
    muyLast = muy
    mux, stdx = norm.fit(mx)
    muy, stdy = norm.fit(my)
    if abs(muxLast) < abs(mux):
        muxAdj = -mux*rotRelax
    else:
        muxAdj = mux*rotRelax

    if abs(muyLast) < abs(muy):
        muyAdj = -muy*rotRelax
    else:
        muyAdj = muy*rotRelax

    ## Save ajdustments in a list in case they are required in future and apply the rotations.

    muxAdjs.append(muxAdj)
    muyAdjs.append(muyAdj)

    rotx = np.array([[1,0,0],[0,math.cos(-muyAdj),-math.sin(-muyAdj)],[0,math.sin(-muyAdj),math.cos(-muyAdj)]])
    XYZ = rotx.dot(np.transpose(XYZ))
    roty = np.array([[math.cos(muxAdj),0,math.sin(muxAdj)],[0,1,0],[-math.sin(muxAdj),0,math.cos(muxAdj)]])
    XYZ = roty.dot(XYZ)
    XYZ = np.transpose(XYZ)

    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]

# ## Sactter plot rotated data

# inDataFig = plt.figure()
# inDataAx = plt.axes(projection='3d')
# inDataAx.scatter(X,Y,Z)
# inDataAx.axis('tight')
# inDataAx.set_xlabel('X [mm]')
# inDataAx.set_ylabel('Y [mm]')
# inDataAx.set_zlabel('Z [mm]')
# inDataAx.set_title('Plot of '+inPath.rsplit('/', 1)[1]+' after reorientation')
# plt.show()

## 2D plot of triangulated data.

fig = plt.figure()
Ax = fig.add_subplot(1, 1, 1)
Ax.triplot(triXY, color='0.7')
Ax.set_xlabel('X [mm]')
Ax.set_ylabel('Y [mm]')
Ax.set_title('2D plot of triangulation')
plt.show()

## 3D plot of triangulated data without flat edge triangles.

fig = plt.figure()
TriXYAx = fig.add_subplot(1, 1, 1, projection='3d')
TriXYAx.plot_trisurf(X, Y, Z, triangles=KeepTriangles, cmap=plt.cm.Spectral)
TriXYAx.axis('tight')
TriXYAx.set_xlabel('X [mm]')
TriXYAx.set_ylabel('Y [mm]')
TriXYAx.set_zlabel('Z [mm]')
TriXYAx.set_title('Plot of '+inPath.rsplit('/', 1)[1]+' after reorientation')
plt.show()

## Plot histograms of X and Y gradient

nbins = round(len(mx)/20)

fig = plt.figure(figsize=plt.figaspect(0.5))
histAx = fig.add_subplot(1, 2, 1)
histAx.hist(mx,bins=nbins)
x_x = np.linspace(mux-3*stdx,mux+3*stdx,1000)
histAx.set_xlabel('Gradient [rad]')
histAx.set_ylabel('Frequency')
histAx.set_title('Histogram of X-components of gradients')
histAx.plot(x_x,stats.norm.pdf(x_x,mux,stdx))

histAx = fig.add_subplot(1, 2, 2)
histAx.hist(my,bins=nbins)
y_y = np.linspace(muy-3*stdy,muy+3*stdy,1000)
histAx.set_xlabel('Gradient [rad]')
histAx.set_ylabel('Frequency')
histAx.set_title('Histogram of Y-components of gradients')
histAx.plot(y_y,stats.norm.pdf(y_y,muy,stdy))
plt.show()

# Define the paraboloid function and do a nonlinear least squares fit.

def func(data, h, k, f, m, n, g):
    return -(((data[:,0]-h)**2+(data[:,1]-k)**2)/(4*f))+m*data[:,0]+n*data[:,1]+g
guess = (1,1,idFoc,1,1,1)
params, pcov = optimize.curve_fit(func, XYZ[:,:2], XYZ[:,2], guess)

# Evaluate the fit at the original target (x,y) coordinates to get the ideal Z coordinates.

ZId = -(((X-params[0])**2+(Y-params[1])**2)/(4*params[2]))+params[3]*X+params[4]*Y+params[5]

# Optional scatter plot of the ideal coordinates.

idFig = plt.figure()
idAx = plt.axes(projection='3d')
idAx.scatter(X,Y,ZId)
idAx.axis('tight')
idAx.set_xlabel('X [mm]')
idAx.set_ylabel('Y [mm]')
idAx.set_zlabel('Z [mm]')
idAx.set_title('Best fit paraboloid with '+str(round(params[2]/1000,2))+' m focal length')
# plt.show()

# Calculate the difference between ideal and measured Z-coordinates at every (x,y) and calculate the mean and standard deviation.

DZ = ZId - Z
muDZ, stdDZ = norm.fit(DZ)

# Reshape the X,Y and Z into m x n matrices for surface plotting. Consider adding a subroutine to automatically find numRows and numCols using dx.

Xgrid = np.reshape(X, (numRows,numCols))
Ygrid = np.reshape(Y, (numRows,numCols))
DZgrid = np.reshape(DZ, (numRows,numCols))

DZDataFig = plt.figure()
DZDataAx = plt.axes(projection='3d')
DZDataAx.plot_surface(Xgrid,Ygrid,DZgrid)
DZDataAx.axis('tight')
DZDataAx.set_xlabel('X [mm]')
DZDataAx.set_ylabel('Y [mm]')
DZDataAx.set_zlabel('Z [mm]')
DZDataAx.set_title('Dz plot of '+inPath.rsplit('/', 1)[1])
plt.show()

## Calculate the ideal surface normal at every point as the mean of the normals of the touching triangles.

XYZId = np.array([X,Y,ZId])
XYZId = np.transpose(XYZId)

# triXYZId = Delaunay(XYZId)

intXYZId = mtri.CubicTriInterpolator(triXY, ZId, trifinder=None)
(mxId,myId) = intXYZId.gradient(triXY.x,triXY.y)

xse = mxId - mx
yse = myId - my
rse = (xse**2+yse**2)**0.5
muxse, stdxse = norm.fit(xse)
muyse, stdyse = norm.fit(yse)
murse, stdrse = norm.fit(rse)

xseFig = plt.figure()
xseAx = plt.axes(projection='3d')
xseAx.scatter(X,Y,xse)
xseAx.axis('tight')
xseAx.set_xlabel('X [mm]')
xseAx.set_ylabel('Y [mm]')
xseAx.set_zlabel('Z [mm]')
xseAx.set_title('X-direction slope error plot of '+inPath.rsplit('/', 1)[1])
plt.show()

yseFig = plt.figure()
yseAx = plt.axes(projection='3d')
yseAx.scatter(X,Y,yse)
yseAx.axis('tight')
yseAx.set_xlabel('X [mm]')
yseAx.set_ylabel('Y [mm]')
yseAx.set_zlabel('Z [mm]')
yseAx.set_title('Y-direction slope error plot of '+inPath.rsplit('/', 1)[1])
plt.show()

nbins  = round(len(mx)/40)

fig = plt.figure(figsize=plt.figaspect(0.5))
histAx = fig.add_subplot(1, 2, 1)
histAx.hist(xse,bins=nbins)
x_xse = np.linspace(muxse-3*stdxse,muxse+3*stdxse,1000)
histAx.set_xlabel('Slope error [rad]')
histAx.set_ylabel('Frequency')
histAx.set_title('Histogram of X-components of slope error')
histAx.plot(x_xse,stats.norm.pdf(x_xse,muxse,stdxse))

histAx = fig.add_subplot(1, 2, 2)
histAx.hist(yse,bins=nbins)
y_yse = np.linspace(muyse-3*stdyse,muyse+3*stdyse,1000)
histAx.set_xlabel('Slope error [rad]')
histAx.set_ylabel('Frequency')
histAx.set_title('Histogram of Y-components of slope error')
histAx.plot(y_yse,stats.norm.pdf(y_yse,muyse,stdyse))
plt.show()

dot = 'end'