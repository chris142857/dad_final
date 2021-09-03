# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:51:48 2021
"""

# PCA for faces

from sklearn.decomposition import PCA
import numpy as np
import dlib
import cv2
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.spatial import procrustes
import warnings
from face.faceAlignment import get_landmark_index_68
from PIL import Image
import matplotlib.pyplot as plt
import pickle

class AppearanceModel():
    def __init__(self, face_images, n_components, output_size, shape_predictor_dir):
        self.shp_pca = PCA(n_components=n_components, whiten=True)
        self.tex_pca = PCA(n_components=n_components, whiten=True)
        self.app_pca = PCA(n_components=n_components, whiten=True)
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_dir)
        self.output_size = output_size
        self.pwa_tf = PiecewiseAffineTransform()
        self.n_components = n_components
        
        src_shapes = []
        # Simple Procrustes alignment
        for img in face_images:
            dets = self.face_detector(img, 1)
            if len(dets) == 0:
                warnings.warn('No face detected!')
                continue
            shp  = self.shape_predictor(img, dets[0])
            lm = np.array([(shp.part(j).x, shp.part(j).y) for j in range(shp.num_parts)])
            src_shapes.append(np.expand_dims(lm, axis=0))
        
        src_shapes = np.concatenate(src_shapes, axis=0)
        dst_shapes = self._align_shapes(src_shapes, (128, 128), output_size)
        mean_shape = np.mean(dst_shapes, axis=0)
        _w, _h = output_size[0], output_size[1]
        dst_corners = np.array([[0, 0], [0, _h-1], [_w-1, 0], [_w-1, _h-1]], dtype=mean_shape.dtype)
        mean_shape_w_c = np.vstack([mean_shape, dst_corners])
        self.mean_shape = mean_shape
        
        # Warp images according to aligned shapes
        wpimgs = []
        for i, img in enumerate(face_images):
            src_shape = src_shapes[i]
            # Add four corners of the image
            left_eye_idx = get_landmark_index_68('LE')
            left_eye = src_shape[left_eye_idx,:]
            right_eye_idx = get_landmark_index_68('RE')
            right_eye = src_shape[right_eye_idx,:]
            left_eye_centre = np.mean(left_eye, axis=0)
            right_eye_centre = np.mean(right_eye, axis=0)
            interoccular = (left_eye_centre + right_eye_centre) / 2
            eyes_dist = np.linalg.norm(right_eye_centre - left_eye_centre)
            minx = int(left_eye_centre[0] - eyes_dist / 0.35 * 0.325)
            maxx = int(right_eye_centre[0] + eyes_dist / 0.35 * 0.325)
            miny = int(interoccular[1] - eyes_dist / 0.3 * 0.35)
            maxy = int(interoccular[1] + eyes_dist / 0.3 * 0.65)            
            corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]], dtype=src_shape.dtype)
            src_shape_w_c = np.vstack([src_shape, corners])
            self.pwa_tf.estimate(mean_shape_w_c, src_shape_w_c)
            wpimg = warp(img, self.pwa_tf, output_shape=(self.output_size[1], self.output_size[0]))
            wpimg[wpimg < 0] = 0
            wpimg[wpimg > 1] = 1
            # _img = Image.fromarray((wpimg * 255).astype(np.uint8))
            # _img.save(f"{i}.jpg")
            # cv2.imwrite(f"{i}.jpg", (wpimg * 255).astype(np.uint8))
            wpimgs.append(wpimg.reshape((1, -1)))
        wpimgs = np.concatenate(wpimgs, axis=0)
        
        self.shp_scores = self.shp_pca.fit_transform(dst_shapes.reshape(dst_shapes.shape[0], -1))
        self.tex_scores = self.tex_pca.fit_transform(wpimgs)
        app_input = np.concatenate([self.shp_scores, self.tex_scores], axis=1)
        self.app_scores = self.app_pca.fit_transform(app_input)
    
    def _align_shapes(self, X, eye_centre, output_size):
        mean_shape = np.mean(X, axis=0)
        aligned_shapes = []
        
        for shp in X:
            _, shp_, _ = procrustes(mean_shape, shp)            
            aligned_shapes.append(np.expand_dims(shp_, 0))
            # plt.scatter(shp_[:, 0], shp_[:, 1])
        
        # plt.show()
        
        aligned_shapes = np.concatenate(aligned_shapes, axis=0)
        mean_shape = np.mean(aligned_shapes, axis=0)
        # Scale and shift the shapes so that they occupy the center of the output images
        left_eye_idx = get_landmark_index_68('LE')
        left_eye = mean_shape[left_eye_idx,:]
        right_eye_idx = get_landmark_index_68('RE')
        right_eye = mean_shape[right_eye_idx,:]
        left_eye_centre = np.mean(left_eye, axis=0)
        right_eye_centre = np.mean(right_eye, axis=0)
        eyes_dist = np.linalg.norm(right_eye_centre - left_eye_centre)
        center_eyes = left_eye_centre + (right_eye_centre - left_eye_centre) / 2
        scale = output_size[0] * 0.35 / eyes_dist
        new_eye_center = np.array([[output_size[0]/2, output_size[1] * 0.35]], dtype=center_eyes.dtype)
        aligned_shapes = (aligned_shapes - center_eyes[np.newaxis, ...]) * scale + \
            new_eye_center[np.newaxis, ...]
        aligned_shapes = np.floor(aligned_shapes)
        
        # for shp in aligned_shapes:
        #     plt.scatter(shp[:, 0], shp[:, 1])
        # plt.show()
        
        return aligned_shapes
    

    def sample(self, n_samples):
        """
        Sample latent representations from the Normal distribution and transform them to images 

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        latents : np.ndarray
            The latent codes with the shape (n_samples, n_latent_dim)
        images : LIST of np.ndarray
            A list of RGB images.

        """
        
        
        # Generate faces from the Normal distribution
        latents = np.random.randn(n_samples, self.n_components)
        images = []
        for i in range(n_samples):
            img = self.decode(latents[i:i+1])
            images.append((img * 255).astype(np.uint8))
        return latents, images
        
    
    def encode(self, img: np.ndarray) -> np.ndarray:
        """
        Transform the input image to its latent representation

        Parameters
        ----------
        img : np.ndarray
            The input image of RGB format.

        Raises
        ------
        ValueError
            Only one single image of RGB type is supported.

        Returns
        -------
        np.ndarray
            The latent representation.

        """
        # Only one image is supported at a time
        if np.ndim(img) != 3:
            raise ValueError("Only one single image of RGB type is supported.")
        dets = self.face_detector(img, 1)
        if len(dets) == 0:
            warnings.warn('No face detected!')
            return None
        shp  = self.shape_predictor(img, dets[0])
        src_shape = np.array([(shp.part(j).x, shp.part(j).y) for j in range(shp.num_parts)])
        
        # Procrustes alignment
        _, aligned_shape, _ = procrustes(self.mean_shape, src_shape)
        # Scale and shift the shapes so that they occupy the center of the output images
        left_eye_idx = get_landmark_index_68('LE')
        left_eye = aligned_shape[left_eye_idx,:]
        right_eye_idx = get_landmark_index_68('RE')
        right_eye = aligned_shape[right_eye_idx,:]
        left_eye_centre = np.mean(left_eye, axis=0)
        right_eye_centre = np.mean(right_eye, axis=0)
        eyes_dist = np.linalg.norm(right_eye_centre - left_eye_centre)
        center_eyes = left_eye_centre + (right_eye_centre - left_eye_centre) / 2
        scale = self.output_size[0] * 0.35 / eyes_dist
        new_eye_center = np.array([[self.output_size[0]/2, self.output_size[1] * 0.35]], dtype=center_eyes.dtype)
        aligned_shape = (aligned_shape - center_eyes) * scale + new_eye_center
        aligned_shape = np.floor(aligned_shape)        
        
        left_eye_idx = get_landmark_index_68('LE')
        left_eye = src_shape[left_eye_idx,:]
        right_eye_idx = get_landmark_index_68('RE')
        right_eye = src_shape[right_eye_idx,:]
        left_eye_centre = np.mean(left_eye, axis=0)
        right_eye_centre = np.mean(right_eye, axis=0)
        interoccular = (left_eye_centre + right_eye_centre) / 2
        eyes_dist = np.linalg.norm(right_eye_centre - left_eye_centre)
        minx = int(left_eye_centre[0] - eyes_dist / 0.35 * 0.325)
        maxx = int(right_eye_centre[0] + eyes_dist / 0.35 * 0.325)
        miny = int(interoccular[1] - eyes_dist / 0.3 * 0.35)
        maxy = int(interoccular[1] + eyes_dist / 0.3 * 0.65)            
        corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]], dtype=src_shape.dtype)
        src_shape = np.vstack([src_shape, corners]) 
        _w, _h = self.output_size[0], self.output_size[1]
        corners = np.array([[0, 0], [0, _h-1], [_w-1, 0], [_w-1, _h-1]], dtype=self.mean_shape.dtype)
        mean_shape = np.vstack((self.mean_shape, corners))
        self.pwa_tf.estimate(mean_shape, src_shape)
        wpimg = warp(img, self.pwa_tf, output_shape=(self.output_size[1], self.output_size[0]))
        wpimg[wpimg < 0] = 0
        wpimg[wpimg > 1] = 1
        # _img = Image.fromarray((wpimg * 255).astype(np.uint8))
        # _img.save(os.path.join("output", "encoded.jpg"))
        # cv2.imwrite("encoded.jpg", (wpimg * 255).astype(np.uint8))
        
        shp_score = self.shp_pca.transform(aligned_shape.reshape(1, -1))
        tex_score = self.tex_pca.transform(wpimg.reshape(1, -1))
        app_input = np.concatenate([shp_score, tex_score], axis=1)
        app_score = self.app_pca.transform(app_input)
        
        return app_score[0]
        
    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Inverse transform from the latent space to the image space

        Parameters
        ----------
        x : np.ndarray
            The latent representation.

        Raises
        ------
        ValueError
            Only one sample is supported.

        Returns
        -------
        wpimg : np.ndarray
            An image of RGB format.

        """
        if np.ndim(x) == 1:
            x = x[np.newaxis, ...]
        elif x.shape[0] > 1:
            raise ValueError("Only one sample is supported")
        shp_tex_scores = self.app_pca.inverse_transform(x)
        shp_scores, tex_scores = np.split(shp_tex_scores, 2, axis=1)
        shp = self.shp_pca.inverse_transform(shp_scores)
        tex = self.tex_pca.inverse_transform(tex_scores)
        shp = shp.reshape(-1, 2)
        tex = tex.reshape(self.output_size[1], self.output_size[0], -1)
        _w, _h = self.output_size[0], self.output_size[1]
        corners = np.array([[0, 0], [0, _h-1], [_w-1, 0], [_w-1, _h-1]], dtype=shp.dtype)
        shp = np.vstack((shp, corners))
        mean_shape = np.vstack((self.mean_shape, corners))
        self.pwa_tf.estimate(shp, mean_shape)
        wpimg = warp(tex, self.pwa_tf, output_shape=(self.output_size[1], self.output_size[0]))
        wpimg[wpimg < 0] = 0
        wpimg[wpimg > 1] = 1
        # _img = Image.fromarray((wpimg * 256).astype(np.uint8))
        # _img.save("decoded.jpg")
        
        return wpimg
    
import os

if __name__ == "__main__":
    input_path = "face_images"
    output_path = "output"
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and (f.endswith('.jpg') or f.endswith('.bmp'))]
    images = []
    
    for file in files:
        img = Image.open(os.path.join(input_path, file), mode='r')
        img = np.array(img)  
        # img = cv2.imread(os.path.join(input_path, file))
        images.append(img)
    
    shape_predictor_dir = 'shape_predictor_68_face_landmarks.dat'
    app_model = AppearanceModel(images, 3, (256, 256), shape_predictor_dir)
    
    # Demonstrate how to save the appearance model
    with open(os.path.join(output_path, "app_model.pkl"), "wb") as f:
        pickle.dump(app_model, f)
    
    # Demonstrate how to load the appearance model
    with open(os.path.join(output_path, "app_model.pkl"), "rb") as f:
        app_model = pickle.load(f)
    
    # Forward transform of a training sample from the image space to the latent space
    latent = app_model.encode(images[0])
    
    # err = np.linalg.norm(latent - app_model.app_scores[0])
    # print(err)

    # Inverse transform of a training sample from the latent space to the image space
    recon = app_model.decode(latent)
    _img = Image.fromarray((recon * 255).astype(np.uint8))
    _img.save(os.path.join(output_path, "recon.jpg"))
    # cv2.imwrite("recon.jpg", (recon * 256).astype(np.uint8))
    
    # Forward transform of an unseen sample from the image space to the latent space
    img = Image.open("42.jpg", mode="r")
    img = np.array(img)
    latent = app_model.encode(img)   
    # Reconstruction of the unseen sample
    recon = app_model.decode(latent)
    _img = Image.fromarray((recon * 255).astype(np.uint8))
    _img.save(os.path.join(output_path, "recon_unseen.jpg"))
    
    # Test sampling
    latents, fake_faces = app_model.sample(n_samples = 5)
    for i, face in enumerate(fake_faces):
        img = Image.fromarray(face)
        img.save(os.path.join(output_path, f"fake_face_{i}.jpg"))
        # cv2.imwrite(f"fake_face_{i}.jpg", face)