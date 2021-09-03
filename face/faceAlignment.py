# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:39:25 2019

"""

import dlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm,colors,rc
import cv2
from IPython import display
import os
 
def get_landmark_index_68(feat):
    LEB = list(range(17,22))
    LE = list(range(36,42))
    LEB_LE = LEB + LE
    REB = list(range(22,27))
    RE = list(range(42,48))
    REB_RE = REB + RE
    Mouth = list(range(48,68))
    Nose = list(range(27,36))
    FacePatch = list(range(0,17))
    
    switch = {'LEB_LE': LEB_LE, 'LEB': LEB, 'LE': LE, 'REB_RE': REB_RE, 
              'REB': REB, 'RE': RE, 'NOSE': Nose, 'MOUTH': Mouth,
              'FP': FacePatch}
    return switch[feat.upper()]

def bbx(box, w, h):
    tbox = [0, 0, 0, 0]
    sbox = [0, 0, 0, 0]
    box_w, box_h = box[2], box[3]
    if box[0] < 0:
        tbox[0] = -box[0]
        box_w -= tbox[0]
    else:
        sbox[0] = box[0]
    if box[1] < 0:
        tbox[1] = -box[1]
        box_h -= tbox[1]
    else:
        sbox[1] = box[1]
    if box[0] + box[2] >= w:
        box_w -= box[0] + box[2] + 1 - w
    if box[1] + box[3] >= h:
        box_h -= box[1] + box[3] + 1 - h
    tbox[2] = tbox[0] + box_w
    tbox[3] = tbox[1] + box_h
    sbox[2] = sbox[0] + box_w
    sbox[3] = sbox[1] + box_h
    return tbox, sbox
    
class FaceNormalizer():
    def __init__(self, output_size, shape_predictor_dir):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_dir)
        self.output_size = output_size
    
    def alignFace2(self, image, left_eye_centre, right_eye_centre):
        left_eye_centre = np.array(left_eye_centre)
        right_eye_centre = np.array(right_eye_centre)
        eyes_dist = np.linalg.norm(right_eye_centre - left_eye_centre)
        center_eyes = left_eye_centre + (right_eye_centre - left_eye_centre) / 2

        face_size_x = int(eyes_dist * 2.)
        if face_size_x < 50: return None # discard tiny faces
 
        # rotate to normalized angle
        d = (right_eye_centre - left_eye_centre) / eyes_dist # normalized eyes-differnce vector (direction)
        a = np.rad2deg(np.arctan2(d[1],d[0])) # angle
        #scale_factor = float(output_size) / float(face_size_x * 2.) # scale to fit in output_size
        # rotation (around center_eyes) + scale transform
        M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1]), a, 1.), [[0,0,1]], axis=0)
        # apply shift from center_eyes to middle of output_size 
        ns = int(eyes_dist/0.2)
        M1 = np.array([[1.,0.,-center_eyes[0]+ns/2.],
                       [0.,1.,-center_eyes[1]+ns/2.],
                       [0,0,1.]])
        # concatenate transforms (rotation-scale + translation)
        M = M1.dot(M)[:2]
        # warp
        try:
            #face = cv2.warpAffine(image, M[:2], (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT)
            face = cv2.warpAffine(image, M[:2], (ns, ns), borderMode=cv2.BORDER_REPLICATE, borderValue=(127,127,127))
        except:
            raise Exception('Error in warpAffine')
        face = cv2.resize(face, self.output_size)
        
        return face        
        
    def alignFace(self, image):
        dets = self.face_detector(image, 1)
        faces = []
        
        for i, d in enumerate(dets):                
            shp = self.shape_predictor(image, d) 
            lm = np.array([(shp.part(j).x, shp.part(j).y) for j in range(shp.num_parts)])
        
            # center and scale face around mid point between eyes
            left_eye_idx = get_landmark_index_68('LE')
            left_eye = lm[left_eye_idx,:]
            right_eye_idx = get_landmark_index_68('RE')
            right_eye = lm[right_eye_idx,:]
            left_eye_centre = np.mean(left_eye, axis=0)
            right_eye_centre = np.mean(right_eye, axis=0)
            eyes_dist = np.linalg.norm(right_eye_centre - left_eye_centre)
            center_eyes = left_eye_centre + (right_eye_centre - left_eye_centre) / 2

            face_size_x = int(eyes_dist * 2.)
            if face_size_x < 50: continue # discard tiny faces
 
            # rotate to normalized angle
            d = (right_eye_centre - left_eye_centre) / eyes_dist # normalized eyes-differnce vector (direction)
            a = np.rad2deg(np.arctan2(d[1],d[0])) # angle
            #scale_factor = float(output_size) / float(face_size_x * 2.) # scale to fit in output_size
            # rotation (around center_eyes) + scale transform
            M = np.append(cv2.getRotationMatrix2D((center_eyes[0], center_eyes[1]), a, 1.), [[0,0,1]], axis=0)
            # apply shift from center_eyes to middle of output_size 
            ns = int(eyes_dist/0.2)
            M1 = np.array([[1.,0.,-center_eyes[0]+ns/2.],
                           [0.,1.,-center_eyes[1]+ns/2.],
                           [0,0,1.]])
            # concatenate transforms (rotation-scale + translation)
            M = M1.dot(M)[:2]
            # warp
            try:
                #face = cv2.warpAffine(image, M[:2], (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT)
                face = cv2.warpAffine(image, M[:2], (ns, ns), borderMode=cv2.BORDER_REPLICATE, borderValue=(127,127,127))
            except:
                continue
#            face_box = np.array([center_eyes[0] - ns*0.5, center_eyes[1] - ns*0.5, ns, ns], dtype=np.int32)
#            face_ = np.ones((face_box[2], face_box[3], 3), dtype=np.float32)*255
#            tbox, sbox = bbx(face_box, face.shape[1], face.shape[0])
#            face_[tbox[1]:tbox[3], tbox[0]:tbox[2], :] = face[sbox[1]:sbox[3], sbox[0]:sbox[2], :]
            face = cv2.resize(face, self.output_size)
            faces.append(face)   
            
        return faces
            
from PIL import Image
import argparse

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='path to dataset folder', type=str, default='../60kind')
    parser.add_argument('--save_dir', default='../60kind_aligned',
            help='path to save overlay images, default=None and do not save images in this case')

    args = parser.parse_args()    
    input_dir = args.input_dir
    save_dir = args.save_dir
    assert os.path.exists(input_dir)

    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    faceAligner = FaceNormalizer(output_size=(256, 256),
                                 shape_predictor_dir='D:\libraries\dlib-master\shape_predictor_68_face_landmarks.dat')

    subpaths = [p for p in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, p))]
    
    for subpath in subpaths:
        subpathfull = os.path.join(input_dir, subpath)
        tsubpathfull = os.path.join(save_dir, subpath)
        if not os.path.exists(tsubpathfull):
            os.mkdir(tsubpathfull)
        else:
            continue
            
        files = [f for f in os.listdir(subpathfull) if os.path.isfile(os.path.join(subpathfull, f)) and (f.endswith('.jpg') or f.endswith('.bmp'))]
    
        for f in files:
            fn,ext = os.path.splitext(f)
            img = Image.open(os.path.join(subpathfull,f), mode='r')
            img = np.array(img)
            if len(img.shape)==0:
                print('{0}: unrecognized image format'.format(f))
                continue                
            if np.ndim(img)==2:
                img = np.repeat(np.expand_dims(img, axis=2), 3, axis=2)
            else:
                img = img[:,:,:3]
            
            faces = faceAligner.alignFace(img)
            
            if faces is None or len(faces)==0:
                continue

            for i,face in enumerate(faces):
                img = Image.fromarray(face)
                #img = img.resize((256,256))
                img.save(os.path.join(tsubpathfull,'{0}_{1}.jpg'.format(fn,i)), mode='RGB')