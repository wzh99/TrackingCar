import numpy as np
import copy as cp

# Tolerance for merging route lines
THETA_THRESH = 0.2
RHO_THRESH = 50

class Point:
    def __init__(self, pos):
        self.pos = tuple(pos[0:2])

class Line:
    def __init__(self, tp):
        # Compute line parameters
        self.pts = (Point(tp[0:2]), Point(tp[2:4]))
        dir = tuple(np.array(self.pts[0].pos) - np.array(self.pts[1].pos))
        len = np.sqrt(dir[0]**2 + dir[1]**2)
        dir = (dir[0]/len, dir[1]/len)
        self.theta = np.arctan2(dir[1], dir[0])
        if self.theta < 0:
            self.theta += np.pi
        norm = (-dir[1], dir[0])
        self.rho = norm[0]*self.pts[0].pos[0] + norm[1]*self.pts[0].pos[1]

        # Compute endpoint parameters
        maxDim = 0
        if np.abs(dir[0]) < np.abs(dir[1]):
            maxDim = 1
        self.p0 = [0, 0]
        minDim = 1 - maxDim
        self.p0[minDim] = self.rho / norm[minDim]
        self.p0 = tuple(self.p0)
        for pt in self.pts:
            pt.t = np.sqrt((pt.pos[0]-self.p0[0])**2 + (pt.pos[1]-self.p0[1])**2)
        if self.pts[0].t > self.pts[1].t:
            self.pts = (self.pts[1], self.pts[0])
        self.len = self.pts[1].t - self.pts[0].t

    def isColinear(self, other) -> bool:
        return np.abs(self.rho - other.rho) < RHO_THRESH and (
            np.abs(self.theta - other.theta) < THETA_THRESH 
                or np.abs(np.abs(self.theta - other.theta) - np.pi) < THETA_THRESH)
        
    def merge(self, other) -> (object, bool):
        if not self.isColinear(other):
            return None, False
        if self.pts[0].t > other.pts[1].t or self.pts[1].t < other.pts[0].t:
            return None, False # no overlapping part
        lineL, lineS = self, other
        if self.len < other.len:
            lineL, lineS = other, self
        if lineL.pts[0].t < lineS.pts[1].t and lineL.pts[1].t > lineS.pts[1].t:
            return lineL, True # lineS is completely inside lineL
        merged = cp.copy(self)
        merged.pts = (min([lineL.pts[0], lineS.pts[0]], key=lambda p: p.t), 
            max([lineL.pts[1], lineS.pts[1]], key=lambda p: p.t))
        return merged, True

    def otherEndpoint(self, pos) -> tuple:
        if pos == self.pts[0].pos:
            return self.pts[1].pos
        elif pos == self.pts[1].pos:
            return self.pts[0].pos
        else:
            raise ValueError("Invalid position.")

def dist(pt1, pt2) -> float:
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

    