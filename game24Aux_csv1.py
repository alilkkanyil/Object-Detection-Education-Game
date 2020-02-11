##########################
# File Name:  game24aux_csv1.py
# Description:   Classes - game and aBlob. An instance of  'game' class initilizes
#                the game, it also creates all the objects of aBlob. Client-Server
#                enabled.
#
# Last Modified:  2/10/2020 - Minor Cleanup
##########################


import os
import cv2 as cv
import time
import numpy as np
import tensorflow as tf
import sys
import random
import math
from utils import label_map_util
from fractions import Fraction
from post24obj import checkTarget, doMath
from copy import deepcopy
sys.path.append("/home/leee/OneDrive/PyChatv2")
import threading
import client #as ct
import json
tfont = cv.FONT_HERSHEY_SIMPLEX
color = {'blue': (255, 0, 0, 255), 'green': (0, 128, 0, 255), 'red': (0, 0, 255, 255), 'black': (
    0, 0, 0, 0), 'white': (255, 255, 255, 255), 'lime': (0, 255, 0, 255), 'pink': (255, 0, 255, 255)}

class game(client.Client):
    # freq=0
    freq = cv.getTickFrequency()
    frame_w, frame_h, frame = 0, 0, None
    fps = 0
    target = 24
    num_boxes = 4
    next_num_box = None
    #max_num_boxes = 2 * (num_boxes - 1) + 1
    num_ops = 4
    ops = ['+', '-', '*', '/']
    origN = [None] * num_boxes  # 4 numbers
    # box_dim=0 #the square in a circle
    c_radius = 0  # the radius of a blob/circle - compute in
    radius_scale = 0.7
    EndGame = False

    # Blob stuffs
    activeBlobs = dict()  # store the current active blob for this loop
    activeBlobLocs = dict()  # 'Repeated': dict(), and ('Num':dict() or 'Op':dict())
    sessionHistory = []  # TBD:a list of lists of the active blob ids of each loop
    repeatedBlobTypes = ['End', 'Restart',  'New', 'Nosol', 'Back']
    repeatedBlobTypeLocs = None
    # selector: call the repeatedBlob type function
    selector = dict()
    # Blobs that are active in each loop - New Game, Restart, End, Hint, Back, etc.
    repeatedBlobs = []
    NumCircles = []  # holding all possible numBlobs
    OpCircles = []  # holding all possible opBlobs
    numNonCmds=0 #Num type if mod 3 =0 or 1.
    active_type = None  # 'Num', 'Op' - type of blobs that are active in the current loop
    # store the two NumBlobs and opBlob to be operated - [num1,num2,op] in postfix form
    BlobstobeOperated = []
    displayTexts = []  # a list for display items for this loop/session
    # Messages
    mesgfontsize = 2
    Mcorners = dict()  # Lower left corners (locations) for Mesgs
    Mesgs = {'No': 'There is no solution!', 'Yes': 'A solution exists.', 'Win': 'You have won!',
             'Lost': 'You have lost!', 'Invalid': 'Wrong Move!'}
    answers = []  # store all  sols in postfix form
    attempts = []  # store all sols that contains the attempts upto the current one
    # attempts=[answers left after 1st attempt, answers left after 2nd attempts, ...]
    ananswerlen = 2 * num_boxes - 1  # len of an answer in postfix form

    def __init__(self, video, serverIP, serverPort):
        client.Client.__init__(self, serverIP, serverPort)
        ###static data###

        self.frame_w = int(video.get(3))  # float
        self.frame_h = int(video.get(4))  # float
        # comp radius
        self.compRadius()
        # repeatedBlobTypeLocs
        h, w, r2 = self.frame_h, self.frame_w, self.c_radius * 2
        w_2, h_2 = w / 2, h / 2
        self.repeatedBlobTypeLocs = [(int(r2 * 1.5), r2), (int(w_2), r2), (int(
            w - r2 * 1.5), r2), (int(w - r2 * 2), int(h - r2)), (r2 * 2, h - r2)]
        for e in self.Mesgs:
            Msize = cv.getTextSize(
                self.Mesgs[e], tfont, self.mesgfontsize, 2)[0]
            self.Mcorners[e] = (int((self.frame_w - Msize[0]) / 2),
                                int((self.frame_h - Msize[1]) / 2))
        # compute the locations where to display actiave blobs
        self.blobLocs()
        self.selector = {'End': self.End, 'Restart': self.Restart,
                         'New': self.New, 'Nosol':self.Nosol, #'Hint': self.Hint,
                         'Back': self.Back}
        ###session data###
        self.active_type = 'Num'  # it begins with Num type

        self.createBlobs()
        # these blobs are always there
        self.activeBlobs['Repeated'] = self.repeatedBlobs
        #overlaod functions
        self.decodeACKWonTime=self.newdecodeACKWonTime
        self.decodeACKWonNosolTime=self.newdecodeACKWonNosolTime
        self.decodeBack=self.newdecodeBack
        self.decodeNew=self.newdecodeNew
        self.decodeRestart=self.newdecodeRestart
        #config
        self.gameconfigdone=False
        self.locsingle=(int(self.frame_w/4), int(self.frame_h/2))
        self.locmultiple=(int(3*self.frame_w/4), int(self.frame_h/2))

        self.singletext="Single"
        self.multipletext="Multiple"
        self.singlesize=cv.getTextSize(self.singletext, tfont, 1, 2)[0]
        self.multiplesize=cv.getTextSize(self.multipletext, tfont, 1, 2)[0]
        self.singlevalCorner = (self.locsingle[0] - self.singlesize[0] // 2,
                     self.locsingle[1] + self.singlesize[1] // 2)
        self.multiplevalCorner = (self.locmultiple[0] - self.multiplesize[0] // 2,
                     self.locmultiple[1] + self.multiplesize[1] // 2)
        #create config blobs - single an multiple
        self.configCircles=[]
        self.configLoc=[]
        self.configvalCorner=[]
        self.configmesg=[]
        self.configCircles.append(aBlob({'isActive': True, 'isLocked': False, \
                                     'blobType': 'config', 'value':self.singletext ,\
                                      'iden': 0, 'cradius': self.c_radius}))
        self.configCircles.append(aBlob({'isActive': True, 'isLocked': False, \
                                     'blobType': 'config', 'value':self.multipletext ,\
                                      'iden': 0, 'cradius': self.c_radius}))
        self.configLoc.append(self.locsingle)
        self.configLoc.append(self.locmultiple)
        print("init loc", self.configLoc)
        self.configvalCorner.append(self.singlevalCorner)
        self.configvalCorner.append(self.multiplevalCorner)
        self.configmesg.append('/Single')
        self.configmesg.append('/Multiple')

    def gameconfighandLock(self,hand_centers, framerate, img):
        for j, b in enumerate(self.configCircles):
            bcenter = self.configLoc[j]
            tparams=b.draw(bcenter, img)
            cv.putText(img, *tparams[0][1])
            cv.putText(img, *tparams[1][1])
            if len(hand_centers) != 0:
                if  b.isHandLocked(bcenter, hand_centers[0], self.c_radius, \
                                   framerate)==True: #locked a blob
                        self.sock.send(bytes(self.configmesg[j], "utf-8"))
                        print("locked config",self.configmesg[j] )
                        self.gameconfigdone=True

        return  # 'Continue'




    def newdecodeACKWonTime(self,param):
        who=json.loads(param[0])
        self.wtime=json.loads(param[1])
        if (who==self.myaddress):
            self.won=True
            tparams=['Repeated',(self.Mesgs['Win'], self.Mcorners['Win'], \
                                 tfont, self.mesgfontsize, color['blue'],  8)]
        else:
            tparams=['Repeated',(self.Mesgs['Lost'], self.Mcorners['Lost'], \
                                 tfont, self.mesgfontsize, color['red'],  8)]
        self.displayTexts.append(tparams)
    def newdecodeACKWonNosolTime(self,param):
        who=json.loads(param[0])
        self.wtime=json.loads(param[1])
        if (who==self.myaddress):
            self.won=True
            tparams=['Repeated',(self.Mesgs['Win'], self.Mcorners['Win'], \
                                 tfont, self.mesgfontsize, color['blue'],  8)]
        else:
            tparams=['Repeated',(self.Mesgs['Lost'], self.Mcorners['Lost'], \
                                 tfont, self.mesgfontsize, color['red'],  8)]
        self.displayTexts.append(tparams)
        print("ACKWonNoSolTime",who, self.wtime, self.won)

    def newdecodeBack(self,param):
        """
        if len(self.N) == 4 and self.numNonCmds%3==0:
            return None
        """
        self.N=json.loads(param[0])
        count=0
        self.active_type="Num"

        """
        c=self.BlobstobeOperated.pop()
        self.numNonCmds=self.numNonCmds-1
        c.cdata['isLocked']=False
        """
        #c=self.BlobstobeOperated.pop()
        self.BlobstobeOperated.clear()
        self.numNonCmds=self.numNonCmds-len(self.BlobstobeOperated)
        for n in self.NumCircles:
            if  count<len(self.N):
                n.cdata['value']=self.N[count]
                n.cdata['isLocked']=False
                n.cdata['active']=True
                n.handCount=0
                count=count+1

        self.updateactiveBlobs()


    def newdecodeNew(self,param):
        #self.N=json.loads(param[0])
        #print("New",self.N)
        self.N=[None]*4
        self.active_type='Num'  #it begins with Num type
        numNonCmds=0
        #self.next_num_box=self.num_boxes
        self.activeBlobs=dict() #store the current active blob for this loop
        self.sessionHistory=[]  #a list of
        self.BlobstobeOperated=[] # store the two NumBlobs to be operated - [num1,num2,op]
        self.displayTexts=[] #
        self.NumCircles.clear()
        self.OpCircles.clear()
        self.repeatedBlobs.clear()
        self.createBlobs()
        #self.updateactiveBlobs()
        self.gameconfigdone=False
        print("reset config", self.gameconfigdone)
    def newdecodeRestart(self,param):
        if (len(self.N)==1 and self.N[0]=="24"):
            return None
        self.N=json.loads(param[0])
        print("Restart",self.N)
        numNonCmds=0
        self.active_type='Num'  #it begins with Num type
        self.next_num_box=self.num_boxes
        self.activeBlobs=dict() #store the current active blob for this loop
        self.sessionHistory=[]  #a list of
        self.BlobstobeOperated=[] # store the two NumBlobs to be operated - [num1,num2,op]
        self.displayTexts=[] #
        self.NumCircles.clear()
        self.OpCircles.clear()
        self.repeatedBlobs.clear()
        self.createBlobs()
        self.updateactiveBlobs()


    def getDim(self):
        return self.frame_w, self.frame_h



    def End(self):
        self.EndGame = True


    def getEndGame(self):
        return self.EndGame

    def Restart(self):
        print('in Restart ...')
        self.sock.send(bytes("/Restart", "utf-8"))

    def New(self):  # Start a new game, clean up all parameters
        print('in New ...')
        self.sock.send(bytes("/New", "utf-8"))

    def Hint(self):
        print('in Hint...')
        None  # TBD

    def Back(self):
        print('in Back ...')
        self.sock.send(bytes("/Back", "utf-8"))


    def Nosol(self):
        print('in Nosol ...')
        self.sock.send(bytes("/Nosol", "utf-8"))

    def loadFrame(self, img):
        self.frame = img

    def getFreq(self):
        return self.freq

    def createBlobs(self):
        # Create the New Game Blob
        ###Static - new state changes ###
        for b in self.repeatedBlobTypes:
            tparams = {'isActive': True, 'isLocked': False, 'blobType': b, 'value': b,
                       'iden': 0, 'cradius': self.c_radius}
            self.repeatedBlobs.append(aBlob(tparams))
        ###Session###
        # Create all Op Blobs
        for i, op in enumerate(self.ops):
            tparams = {'isActive': True, 'isLocked': False, 'blobType': 'Op', 'value': self.ops[i],
                       'iden': i, 'cradius': self.c_radius}
        # True, 'Op', self.ops[i], self.c_radius,i)
            self.OpCircles.append(aBlob(tparams))
        # Create all the Num Blobs
        for i in range(self.num_boxes):
            tparams = {'isActive': True, 'isLocked': False, 'blobType': 'Num', 'value': self.N[i],
                       'iden': i, 'cradius': self.c_radius}
            # True, 'Num', self.N[i], self.c_radius,i)
            self.NumCircles.append(aBlob(tparams))

    def blobLocs(self):  # Compute the locations of all blobs
        tmp = []
        # 'Repeated' blobs such as 'Restart', ...
        for k, b in enumerate(self.repeatedBlobTypes):
            tmp.append(self.repeatedBlobTypeLocs[k])
        self.activeBlobLocs['Repeated'] = {len(self.repeatedBlobTypes): tmp}
        # session Blob locations
        # for 'Num': find the centers for .. , 4, 3 and 2 numbers
        tmp = dict()
        for i in range(1, self.num_boxes + 1):
            tmp[i] = [(int((2 * j + 1) / (2 * i + 1) * (self.frame_w) + self.c_radius / 2),
                       int(self.frame_h * 0.65)) for j in range(i)]
        self.activeBlobLocs['Num'] = tmp
        # for 'Op': find the centers for 4 operaions
        self.activeBlobLocs['Op'] = {self.num_ops: [(int((2 * i + 1) / (2 * self.num_ops + 1) *
                                                         (self.frame_w) + self.c_radius / 2), int(self.frame_h * 0.4))
                                                    for i in range(self.num_ops)]}

    def printBlobs(self):
        for e in self.OpCircles:
            e.bPrint()
        for e in self.NumCircles:
            e.bPrint()

    def printActiveBlobs(self):
        for e in self.activeBlobs:
            e.bPrint()

    def printToBeOperatedBlobs(self):
        for e in self.BlobstobeOperated:
            e.bPrint()

    # update activeBlob list of the loop
    # First len(repeatedBlobs) are Repeated type, the rest are either Num or Op type
    def updateactiveBlobs(self):

        # E.g. activeBlobs={'Repeated':[Some blobs ...],'Num':[....]}
        ###Static-repeated list###
        self.activeBlobs = dict()
        self.activeBlobs['Repeated'] = self.repeatedBlobs
        ###Session-Num or Op list###
        tmp = []

        if self.active_type == 'Num':
            #print("circle - N", len(self.NumCircles), self.N)
            count=0
            for n in self.NumCircles:
                if  count<len(self.N):
                    n.cdata['value']=self.N[count]
                    #n.cdata['isLocked']=False
                    count=count+1
                    tmp.append(n)
                    n.cdata['active']=True
                else:
                    n.cdata['active']=False
            self.activeBlobs['Num'] = tmp
        elif self.active_type == 'Op':
            self.activeBlobs['Op'] = self.OpCircles
        else:
            None




    # check if any activeBlob got locked
    # for now we use only the first hand
    def handlockedBlob(self, hand_centers, framerate):
        btype = None
        if len(hand_centers) == 0:
            return  # no hands
        for atype, activeB in self.activeBlobs.items():
            lab = len(activeB)
            count=0
            for j, b in enumerate(activeB):
                if b.getLock() == False and b.getActive() == True:
                    bcenter = self.activeBlobLocs[atype][lab][j]
                    t_lock = b.isHandLocked(
                        bcenter, hand_centers[0], self.c_radius, framerate)
                    if t_lock == True:
                        btype = b.getType()
                        # for Repeated type - 'New','End', etc.
                        if (btype in self.repeatedBlobTypes):
                            self.selector[btype]()  # call New(), End(), etc.
                            return  # 'Continue'
                        else:
                            b.cdata['loc']=str(j)
                            count=count+1
                            self.BlobstobeOperated.append(b)
                            self.numNonCmds=self.numNonCmds+1
                        break
        lb = len(self.BlobstobeOperated)
        if lb == 3:
            # self.compSelectedBlobs(self.BlobstobeOperated)
            #self.checkSelectedBlobs(self.BlobstobeOperated)
            bv=''
            #clear blob's lock as well
            for b in self.BlobstobeOperated:
                print("blob ",b.cdata['value'])
                bv=bv+b.cdata['loc']+' '
                b.cdata['isLocked']=False
            self.sock.send(bytes(bv, "utf-8"))

            self.BlobstobeOperated = []
            self.active_type = 'Num'
        elif lb == 2:
            self.active_type = 'Op'
        return  # 'Continue'

    # draw active Blobs for display, and prepare their and others texts
    def processactiveBlobs(self):
        # activeB=[some blobs ....]
        for atype, activeB in self.activeBlobs.items():
            lab = len(activeB)
            for j, b in enumerate(activeB):
                tmp = b.draw(self.activeBlobLocs[atype][lab][j], self.frame)
                self.displayTexts.extend(tmp)
        tokeep = []
        for i, d in enumerate(self.displayTexts):
            cv.putText(self.frame, *(d[1]))
            if d[0] == 'Repeated':
                tokeep.append(d)
        self.displayTexts = tokeep

    def box_normal_to_pixel(self, box, dim):
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width),
                     int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    def compRadius(self):
        box_dim = int(max(self.frame_h, self.frame_w) /
                      (2 * self.num_boxes + 1))
        self.c_radius = math.ceil(self.radius_scale * box_dim / math.sqrt(2))

# Blobs -- number circles, operator circles, etc.


class aBlob():
    cdata = {'isActive': None, 'isLocked': None, 'blobType': None,
             'value': None, 'iden': None, 'cradius': None,'loc':None}
    textsize = None
    valCorner = None
    handCount = 0
    t1 = 0
    t2 = 0
    lockscale = 1.8  # hand lock senitivity > 1
    # This parents  is None if it is the original blob containing one of the numbers
    # ANd the 'isActive' is True.
    fontsize = 1
    parents = None  # history - [numBlob1, opBlob, numBlob2]

    def __init__(self, params):  # sactive, blobtype, val, radius,iden):
        self.handCount = 0
        self.t1 = 0
        self.t2 = 0
        self.updateBlob(params)
        self.textsize = cv.getTextSize(
            self.cdata['value'], tfont, self.fontsize, 2)[0]

    def updateParents(self, ancestor):
        self.parents = ancestor

    def updateBlob(self, params):  # params dict of data
        t = deepcopy(self.cdata)
        t.update(params)
        self.cdata = t
        if 'value' in params:
            self.textsize = cv.getTextSize(
                str(self.cdata['value']), tfont, self.fontsize, 2)[0]

    def getParents(self):
        return self.parents

    def getLock(self):
        return self.cdata['isLocked']

    def getVal(self):
        return self.cdata['value']

    def getData(self):
        return self.cdata

    def getId(self):
        return self.cdata['iden']

    def getActive(self):
        return self.cdata['isActive']

    def getRadius(self):
        return self.cdata['cradius']

    def getType(self):
        return self.cdata['blobType']

    # Check if this blob is locked on by a hand
    def isHandLocked(self, c_center, h_center, rad, framerate):
        if ((c_center[0] - h_center[0])**2) + ((c_center[1] - h_center[1])**2) < (rad**2):
            self.handCount = self.handCount + 1
            if (self.handCount >= self.lockscale * framerate):
                if (self.getType() == 'Num'):  # Don't update this for 'Op' or any 'Repeated' types
                    self.cdata['isLocked'] = True
                self.handCount = 0
                return True
        return False

    def draw(self, location, img):
        cFull = 2
        if (self.getLock() == True):
            cFull = -1
        cv.circle(img, location, self.getRadius(), color['lime'], cFull)
        valCorner = (location[0] - self.textsize[0] // 2,
                     location[1] + self.textsize[1] // 2)
        #cv.putText(img, self.getVal(), valCorner, tfont,1.5, color['black'],  5)
        sv = self.getVal()
        tparams1 = ['Once', (sv, valCorner, tfont,
                             self.fontsize, color['black'],  5)]
        tparams2 = ['Once', (sv, valCorner, tfont,
                             self.fontsize, color['white'],  1)]
        return [tparams1, tparams2]
        # return tparams2

    def bPrint(self):
        print(self.cdata)
