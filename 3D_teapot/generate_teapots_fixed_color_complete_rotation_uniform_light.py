__author__ = 'pol'

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import glfw
import generative_models
from utils import *
import OpenGL.GL as GL
from utils import *
import h5py
import cv2
import os
plt.ion()
from OpenGL import contextdata
from tqdm import tqdm

#__GL_THREADED_OPTIMIZATIONS

#Main script options:

glModes = ['glfw','mesa']
glMode = glModes[0]

np.random.seed(1)

width, height = (64, 64)
numPixels = width*height
shapeIm = [width, height,3]
win = -1
clip_start = 0.05
clip_end = 10
frustum = {'near': clip_start, 'far': clip_end, 'width': width, 'height': height}

if glMode == 'glfw':
    #Initialize base GLFW context for the Demo and to share context among all renderers.
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.DEPTH_BITS,32)
    glfw.window_hint(glfw.VISIBLE, GL.GL_FALSE)
    win = glfw.create_window(width, height, "Demo",  None, None)
    glfw.make_context_current(win)

else:
    from OpenGL.raw.osmesa._types import *
    from OpenGL.raw.osmesa import mesa

winShared = None

gtCamElevation = np.pi/3
gtCamHeight = 0.3 #meters

chLightAzimuthGT = ch.Ch([0])
chLightElevationGT = ch.Ch([np.pi/3])
chLightIntensityGT = ch.Ch([1])
chGlobalConstantGT = ch.Ch([0.5])

chCamElGT = ch.Ch([gtCamElevation])
focalLenght = 35 ##milimeters
chCamFocalLengthGT = ch.Ch([35/1000])

#Move camera backwards to match the elevation desired as it looks at origin:
# bottomElev = np.pi/2 - (gtCamElevation + np.arctan(17.5 / focalLenght ))
# ZshiftGT =  ch.Ch(-gtCamHeight * np.tan(bottomElev)) #Move camera backwards to match the elevation desired as it looks at origin.

ZshiftGT =  ch.Ch([-0.5])

# Baackground cube - add to renderer by default.
verticesCube, facesCube, normalsCube, vColorsCube, texturesListCube, haveTexturesCube = getCubeData()

uvCube = np.zeros([verticesCube.shape[0],2])

chCubePosition = ch.Ch([0, 0, 0])
chCubeScale = ch.Ch([10.0])
chCubeAzimuth = ch.Ch([0])
chCubeVCColors = ch.Ch(np.ones_like(vColorsCube) * 1) #white cube
v_transf, vn_transf = transformObject([verticesCube], [normalsCube], chCubeScale, chCubeAzimuth, chCubePosition)

v_scene = [v_transf]
f_list_scene = [[[facesCube]]]
vc_scene = [[chCubeVCColors]]
vn_scene = [vn_transf]
uv_scene = [[uvCube]]
haveTextures_list_scene = [haveTexturesCube]
textures_list_scene = [texturesListCube]

#Example object 1: teapot

chPositionGT = ch.Ch([0, 0, 0.])
# chPositionGT = ch.Ch([-0.23, 0.36, 0.])
chScaleGT = ch.Ch([1.0, 1.0, 1.0])
chAzimuthGT = ch.Ch([np.pi/3])
chAxGT = ch.Ch([np.pi/3])
chAz2GT = ch.Ch([np.pi/2])
chVColorsGT = ch.Ch([0.3, 0.5, 0.9])

import shape_model
# %% Load data
#You can get the teapot data from here: https://drive.google.com/open?id=1JO5ZsXHb_KTsjFMFx7rxY0YVAwnM3TMY
filePath = 'data/teapotModel.pkl'
teapotModel = shape_model.loadObject(filePath)
faces = teapotModel['faces']

# %% Sample random shape Params
latentDim = np.shape(teapotModel['ppcaW'])[1]
shapeParams = np.zeros(latentDim)
chShapeParams = ch.Ch(shapeParams.copy())

meshLinearTransform = teapotModel['meshLinearTransform']
W = teapotModel['ppcaW']
b = teapotModel['ppcaB']

chVertices = shape_model.VerticesModel(chShapeParams=chShapeParams, meshLinearTransform=meshLinearTransform, W=W, b=b)
chVertices.init()

chVertices = ch.dot(geometry.RotateZ(-np.pi/2)[0:3, 0:3], chVertices.T).T

smFaces = [[faces]]
smVColors = [chVColorsGT * np.ones(chVertices.shape)]
smUVs = ch.Ch(np.zeros([chVertices.shape[0],2]))
smHaveTextures = [[False]]
smTexturesList = [[None]]

chVertices = chVertices - ch.mean(chVertices, axis=0)

chVertices = chVertices * 0.09
smCenter = ch.array([0, 0, 0.1])

smVertices = [chVertices]
chNormals = shape_model.chGetNormals(chVertices, faces)
smNormals = [chNormals]

center = smCenter
UVs = smUVs
v = smVertices
vn = smNormals
Faces = smFaces
VColors = smVColors
HaveTextures = smHaveTextures
TexturesList = smTexturesList

v_transf, vn_transf = transformObjectFull(v, vn, chScaleGT, chAzimuthGT, chAxGT, chAz2GT, chPositionGT)

vc_illuminated = computeGlobalAndDirectionalLighting(vn_transf, VColors, chLightAzimuthGT, chLightElevationGT, chLightIntensityGT, chGlobalConstantGT)

v_scene += [v_transf]
f_list_scene += [smFaces]
vc_scene += [vc_illuminated]
vn_scene += [vn_transf]
uv_scene += [UVs]
haveTextures_list_scene += [HaveTextures]
textures_list_scene += [TexturesList]

#COnfigure lighting
lightParamsGT = {'chLightAzimuth': chLightAzimuthGT, 'chLightElevation': chLightElevationGT, 'chLightIntensity': chLightIntensityGT, 'chGlobalConstant':chGlobalConstantGT}

c0 = width/2  #principal point
c1 = height/2  #principal point
#a1 = 3.657  #Aspect ratio / mm to pixels
a1 = 3
#a2 = 3.657  #Aspect ratio / mm to pixels
a2 = 3

cameraParamsGT = {'Zshift':ZshiftGT, 'chCamEl': chCamElGT, 'chCamFocalLength':chCamFocalLengthGT, 'a':np.array([a1,a2]), 'width': width, 'height':height, 'c':np.array([c0, c1])}

#Create renderer object
renderer = createRenderer(glMode, cameraParamsGT, v_scene, vc_scene, f_list_scene, vn_scene, uv_scene, haveTextures_list_scene,
                               textures_list_scene, frustum, None)
# Initialize renderer
renderer.overdraw = True
renderer.nsamples = 8
renderer.msaa = True  #Without anti-aliasing optimization often does not work.
renderer.initGL()
renderer.initGLTexture()
renderer.debug = False
winShared = renderer.win


print("Creating Ground Truth")

trainAzsGT = np.array([])
trainObjAzsGT = np.array([])
trainElevsGT = np.array([])
trainLightAzsGT = np.array([])
trainLightElevsGT = np.array([])
trainLightIntensitiesGT = np.array([])
trainVColorGT = np.array([])
trainIds = np.array([], dtype=np.uint32)
trainAmbientIntensityGT = np.array([])
trainShapeModelCoeffsGT = np.array([]).reshape([0,latentDim])


gtDtype = [('trainIds', trainIds.dtype.name),
           ('trainAzsGT', trainAzsGT.dtype.name),
           ('trainElevsGT', trainElevsGT.dtype.name),
           ('trainLightAzsGT', trainLightAzsGT.dtype.name),
           ('trainLightElevsGT', trainLightElevsGT.dtype.name),
           ('trainLightIntensitiesGT', trainLightIntensitiesGT.dtype.name),
           ('trainAmbientIntensityGT', trainAmbientIntensityGT.dtype),
           ('trainVColorGT', trainVColorGT.dtype.name, (3,) ),
           ('trainShapeModelCoeffsGT', trainShapeModelCoeffsGT.dtype, (latentDim,)),]


#groundTruth = np.array([], dtype = gtDtype)
#groundTruthFilename = gtDir + 'groundTruth.h5'
#gtDataFile = h5py.File(groundTruthFilename, 'a')

#gtDataset = gtDataFile.create_dataset(prefix, data=groundTruth, maxshape=(None,))

train_data_path = "./teapot_dataset_fixed_color_complete_rotation_uniform_light/train_data/"
train_data_image_path = os.path.join(train_data_path, "images")
train_data_npz_path = os.path.join(train_data_path, "data.npz")

metric_data_path = "./teapot_dataset_fixed_color_complete_rotation_uniform_light/metric_data/"
metric_data_image_path = os.path.join(metric_data_path, "images")
metric_data_npz_path = os.path.join(metric_data_path, "data.npz")

if not os.path.exists(train_data_image_path):
    os.makedirs(train_data_image_path)
if not os.path.exists(metric_data_image_path):
    os.makedirs(metric_data_image_path)

np.random.seed(1) #Set a seed for reproducibility.

def generate_random_direction(size_):
    gaussian_sample = np.random.normal(0, 1, size=(size_, 3))
    x = gaussian_sample[:, 0]
    y = gaussian_sample[:, 1]
    z = gaussian_sample[:, 2]
    norm = np.sqrt((x**2) + (y**2) + (z**2))
    x, y, z = x / norm, y / norm, z / norm
    t1 = np.arctan(np.sqrt((x**2) + (y**2)) / z)
    t1[t1 < 0] += np.pi
    t2 = np.arctan2(x, y)
    t2[t2 < 0] += np.pi * 2
    return t1, t2 ## \beta, \alpha

def generate_random_latents(size_):
    latents = np.zeros((size_, 5))
    latents[:, 0] = np.random.uniform(0.0, 2 * np.pi, size_) ## \gamma
    t1, t2 = generate_random_direction(size_)
    latents[:, 1] = t1 ## \beta
    latents[:, 2] = t2 ## \alpha
    t3, t4 = generate_random_direction(size_)
    latents[:, 3] = t4 ## chLightAzGTVals
    latents[:, 4] = t3 ## chLightElGTVals
    return latents

num_train_data = 200000

train_data_latents = generate_random_latents(num_train_data)
train_data_imgs = []

for train_i in tqdm(range(num_train_data)):
    #chAzGTVals = np.mod(np.random.uniform(0, np.pi, 1) - np.pi / 2, 2 * np.pi)
    chElGTVals = 0.0
    #chLightAzGTVals = np.random.uniform(0, 2 * np.pi, 1)
    chLightAzGTVals = train_data_latents[train_i, 3]
    #chLightElGTVals = np.random.uniform(0, np.pi / 2, 1)
    chLightElGTVals = train_data_latents[train_i, 4]
    #chAmbientIntensityGTVals = np.random.uniform(0.4, 0.6)
    chAmbientIntensityGTVals = 0.5
    #chLightIntensityGTVals = np.random.uniform(0.8, 1.0)
    chLightIntensityGTVals = 0.9
    #chVColorsGTVals = np.random.uniform(0.0, 1.0, [1, 3])
    #shapeParamsVals = np.random.randn(latentDim)
    shapeParamsVals = np.asarray([0,0,0,0,0,0,0,0,0,0])

    chAzimuthGT[:] = train_data_latents[train_i, 0]
    chAxGT[:] = train_data_latents[train_i, 1]
    chAz2GT[:] = train_data_latents[train_i, 2]
    chCamElGT[:] = chElGTVals
    chLightAzimuthGT[:] = chLightAzGTVals
    chLightElevationGT[:] = chLightElGTVals
    chLightIntensityGT[:] = chLightIntensityGTVals
    chGlobalConstantGT[:] = chAmbientIntensityGTVals
    #chVColorsGT[:] = chVColorsGTVals
    chShapeParams[:] = shapeParamsVals

    image = renderer.r.copy()

    #cv2.imwrite(os.path.join(train_data_image_path, 'im' + str(train_i) + '.jpeg'), 255 * image[:, :, :],
    #            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #np.save(os.path.join(train_data_image_path, 'im' + str(train_i) + '.npy'), image)
    train_data_imgs.append(image)

    #gtDataset.resize(gtDataset.shape[0] + 1, axis=0)
    #gtDataset[-1] = np.array([(train_i,
    #                           chAzGTVals,
    #                           chElGTVals,
    #                           chLightAzGTVals,
    #                           chLightElGTVals,
    #                           chLightIntensityGTVals,
    #                           chAmbientIntensityGTVals,
    #                           chVColorsGTVals,
    #                           shapeParamsVals,
    #                           )], dtype=gtDtype)

    #gtDataFile.flush()
    #print("Generated " + str(train_i) + " GT instances.")
train_data_imgs = np.stack(train_data_imgs, axis=0)
np.savez(train_data_npz_path, imgs=train_data_imgs, latents=train_data_latents)
#np.savez(train_data_npz_path, latents=train_data_latents)
print("Finished generating train data")

L = 100
M = 500

metric_data_imgs = []
metric_data_labels = []
for i in tqdm(range(M)):
    fixed_latent_id = i % 5
    latents = generate_random_latents(L)
    latents[:, fixed_latent_id] = latents[0, fixed_latent_id]

    sub_metric_data_imgs = []
    for j in tqdm(range(L)):
        #chAzGTVals = np.mod(np.random.uniform(0, np.pi, 1) - np.pi / 2, 2 * np.pi)
        #chElGTVals = np.random.uniform(0.05, np.pi / 2, 1)
        chElGTVals = 0.0
        #chLightAzGTVals = np.random.uniform(0, 2 * np.pi, 1)
        chLightAzGTVals = latents[j, 3]
        #chLightElGTVals = np.random.uniform(0, np.pi / 2, 1)
        chLightElGTVals = latents[j, 4]
        #chAmbientIntensityGTVals = np.random.uniform(0.4, 0.6)
        chAmbientIntensityGTVals = 0.5
        #chLightIntensityGTVals = np.random.uniform(0.8, 1.0)
        chLightIntensityGTVals = 0.9
        #chVColorsGTVals = np.random.uniform(0.0, 1.0, [1, 3])
        #shapeParamsVals = np.random.randn(latentDim)
        shapeParamsVals = np.asarray([0,0,0,0,0,0,0,0,0,0])

        chAzimuthGT[:] = latents[j, 0]
        chAxGT[:] = latents[j, 1]
        chAz2GT[:] = latents[j, 2]
        chCamElGT[:] = chElGTVals
        chLightAzimuthGT[:] = chLightAzGTVals
        chLightElevationGT[:] = chLightElGTVals
        chLightIntensityGT[:] = chLightIntensityGTVals
        chGlobalConstantGT[:] = chAmbientIntensityGTVals
        #chVColorsGT[:] = chVColorsGTVals
        chShapeParams[:] = shapeParamsVals

        image = renderer.r.copy()

        #cv2.imwrite(os.path.join(metric_data_image_path, 'im{}_{}.jpeg'.format(i, j)), 255 * image[:, :, :],
        #            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #np.save(os.path.join(metric_data_image_path, 'im{}_{}.npy'.format(i, j)), image)
        sub_metric_data_imgs.append(image)

        #gtDataset.resize(gtDataset.shape[0] + 1, axis=0)
        #gtDataset[-1] = np.array([(train_i,
        #                           chAzGTVals,
        #                           chElGTVals,
        #                           chLightAzGTVals,
        #                           chLightElGTVals,
        #                           chLightIntensityGTVals,
        #                           chAmbientIntensityGTVals,
        #                           chVColorsGTVals,
        #                           shapeParamsVals,
        #                           )], dtype=gtDtype)

        #gtDataFile.flush()
        #print("Generated " + str(train_i) + " GT instances.")
    metric_data_imgs.append(np.stack(sub_metric_data_imgs, axis=0))
    metric_data_labels.append(fixed_latent_id)

metric_data_imgs = np.stack(metric_data_imgs, axis=0)
metric_data_labels = np.asarray(metric_data_labels, dtype=np.int32)

np.savez(metric_data_npz_path, imgs=metric_data_imgs, labels=metric_data_labels)
#np.savez(metric_data_npz_path, labels=metric_data_labels)
print("Finished generating metric data")

exit()
plt.figure()
plt.title('GT object')
plt.imshow(renderer.r)
plt.show(0.1)



#Clean up.
renderer.makeCurrentContext()
renderer.clear()
contextdata.cleanupContext(contextdata.getContext())
# glfw.destroy_window(renderer.win)
del renderer