#%%
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa
import PIL
import numpy as np
import pickle
import os
import pandas as pd
import kerastuner as kt
import argparse
#%%
parser = argparse.ArgumentParser()
# flags for script
parser.add_argument("--host", type=str, default='local')
parser.add_argument("--step", type=str, default='train')
parser.add_argument("--debug", type=int, default=0)
# flags for general structure
parser.add_argument("--part", type=str, default='name')
parser.add_argument("--core", type=str, default='efficientnetb4')
parser.add_argument("--aug", type=int, default=1)
parser.add_argument("--training_mode", type=str, default="kt")
# arguments for manual training
parser.add_argument("--global_pooling", type=str, default="avg")
parser.add_argument("--freeze", type=str, default="none")
parser.add_argument("--nnoutput", type=str, default="dense")
parser.add_argument("--base_lr", type=float, default=5e-5)
# flags for training
parser.add_argument("--batch_size", type=int, default=8)
# parser.add_argument("--init_with", type=str, default="local_imagenet")
# misc flags, should normally not be changed
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--im_size", type=int, default=320)

FLAGS = parser.parse_args()

VERSION=1
DEBUG = (FLAGS.debug != 0) & (FLAGS.host=="local")
if DEBUG:
    BATCH_SIZE = 2
    EPOCHS = 100
else:
    BATCH_SIZE = FLAGS.batch_size
    EPOCHS = 100
IM_SIZE = FLAGS.im_size
CORE = FLAGS.core

# Other hyperparameters

print("Init cells segmentation w/ u-net script")
print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

if FLAGS.host=="local":
    data_path = r"C:/Users/Jean-Baptiste/OneDrive/ENSAE/2A/CHU/Prediction_soustype/data"
    output_path = r"C:/Users/Jean-Baptiste/OneDrive/ENSAE/2A/CHU/Prediction_soustype/models"
elif FLAGS.host=="jeanzay":
    data_path = "/gpfsdswork/projects/rech/ild/uqk67mt/vexas/data/segmented"
    output_path = "/gpfsdswork/projects/rech/ild/uqk67mt/vexas/out"
    
def getModelName():
    if VERSION==1:
        # if FLAGS.part=="reg":
        if FLAGS.training_mode == "kt":
            return "vx1_scell_{}_{}_kt".format(CORE,IM_SIZE)
        elif FLAGS.training_mode == "man":
            return "vx1_scell_{}_{}_man_{}_{}_{}_{:.0e}".format(CORE,IM_SIZE,
                                                                FLAGS.global_pooling,
                                                                FLAGS.freeze,
                                                                FLAGS.nnoutput,
                                                                FLAGS.base_lr)
        # elif FLAGS.part=="name":
        #     if FLAGS.training_mode == "kt":
        #         return "vx1_partname_scell_{}_{}_kt".format(CORE,IM_SIZE)
        #     elif FLAGS.training_mode == "man":
        #         return "vx1_partname_scell_{}_{}_man_{}_{}_{}_{:.0e}".format(CORE,IM_SIZE,
        #                                                                      FLAGS.global_pooling,
        #                                                                      FLAGS.freeze,
        #                                                                      FLAGS.nnoutput,
        #                                                                      FLAGS.base_lr)
MODEL_NAME = getModelName()
    
# TODO EfficientNet

# TODO certaines cellules ont des "trous"
# revoir la segmentation des cellules ?


# %%

def moderate_augmentation(images, seed=None):
    # 'Apply data augmentation'
    if seed is not None:
        ia.seed(seed)
    often = lambda aug: iaa.Sometimes(0.9, aug)
    sometimes = lambda aug: iaa.Sometimes(0.6, aug)
    seldom = lambda aug: iaa.Sometimes(0.3, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # sometimes(iaa.CropAndPad(
            #     percent=(-0.05, 0.1),
            #     pad_mode=ia.ALL,
            #     pad_cval=(0, 255)
            # )),
            often(iaa.Affine(
                rotate=(-180, 180), # rotate by -45 to +45 degrees
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                # rotate=(-180, 180), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                mode='constant'
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            seldom(iaa.SomeOf((0, 5),
                [
                    # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    # iaa.SimplexNoiseAlpha(iaa.OneOf([
                    #     iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    #     iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    # ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                    # iaa.OneOf([
                    #     iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                    #     iaa.CoarseDropout((0.01, 0.05), size_percent=(0.01, 0.05), per_channel=0.2),
                    # ]),
                    # iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    # iaa.OneOf([
                    #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    #     iaa.FrequencyNoiseAlpha(
                    #         exponent=(-4, 0),
                    #         first=iaa.Multiply((0.5, 1.5), per_channel=True),
                    #         second=iaa.LinearContrast((0.5, 2.0))
                    #     )
                    # ]),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    # iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            ))
        ],
        random_order=True
    )
    return seq.augment_images(images)

def light_augmentation(images, seed=None):
    if seed is not None:
        ia.seed(seed)
    # 'Apply data augmentation'
    # print('Applying light augmentation')
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 20% of all images
            iaa.Affine(
                rotate=(-180, 180), # rotate by -45 to +45 degrees
            ),
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                # rotate=(-180, 180), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 0), # if mode is constant, use a cval between 0 and 255
                mode='constant'
            ))
        ],
        random_order=True
    )
    return seq.augment_images(images)

def adjust(images, width, height):
    seq = iaa.Sequential(
        [
            iaa.CropToFixedSize(width, height, position='center'),
            iaa.PadToFixedSize(width, height, position='center')
        ],
        random_order=False
    )
    return seq.augment_images(images)

class ImgAugDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images, y, batch_size, width, height, preprocess_function, shuffle=False, augment=0, full_batches_only=True, seed=42):
        print("Custom imgaug generator: found {} images belonging to {} classes.".format(len(images),y.shape[1]))
        self.images_paths        = images              # array of image paths
        self.labels              = y                   # array of labels (one hot encoded)
        self.width               = width               # image dimensions
        self.height              = height              # image dimensions
        self.batch_size          = batch_size          # batch size
        self.shuffle             = shuffle             # shuffle bool
        self.augment             = augment             # augment data bool
        self.preprocess_function = preprocess_function
        self.full_batches_only   = full_batches_only
        self.seed                = seed
        self.rng                 = np.random.RandomState(seed)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.full_batches_only:
            return int(np.floor(len(self.images_paths) / self.batch_size))
        else:
            return int(np.ceil(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #if False:
        #    indexes = np.arange(len(images_paths))
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : min((index + 1) * self.batch_size, len(self.indexes))]

        # select data and load images
        labels = np.array([self.labels[k] for k in indexes])
        # images = [cv2.imread(self.images_paths[k], cv2.COLOR_BGR2RGB) for k in indexes]
        # images = [cv2.imread(self.images_paths[k]) for k in indexes]
        # images = [cv2.cvtColor(cv2.imread(self.images_paths[k]), cv2.COLOR_BGR2RGB) for k in indexes]
        images = [np.array(PIL.Image.open(self.images_paths[k])) for k in indexes]
        
        # preprocess and augment data
        if self.augment == 1:
            images = light_augmentation(images, self.seed)
        elif self.augment == 2:
            images = moderate_augmentation(images, self.seed)
        self.seed += 1
            
        images = adjust(images, self.width, self.height)
        
        # if self.preprocess_function is not None:
        images = np.array([self.preprocess_function(img) for img in images])
        return images, labels


# %%

KEEP_GROUPS = ['LAL B1','LAL B2', 'LAL B3']
N_CLASSES = len(KEEP_GROUPS)

input_data = {}
for part in ('train','valid','test'):
    # if FLAGS.part=="reg":
    part_file_name = "part_{}.pkl".format(part)
    # elif FLAGS.part=="name":
    #     part_file_name = "part_name_{}.pkl".format(part)
    with open(os.path.join(data_path,part_file_name), 'rb') as file_pi:
       tmp_meta = pickle.load(file_pi)
    
    input_data[part]={}
    input_data[part]['images'] = [os.path.join(data_path,meta['PATH']) for meta in tmp_meta if meta['GROUP'] in KEEP_GROUPS]
    tmp_y = [meta['GROUP'] for meta in tmp_meta if meta['GROUP'] in KEEP_GROUPS]
    tmp_y = pd.get_dummies(tmp_y) # one hot encode
    if tmp_y.shape[1] == len(KEEP_GROUPS):
        tmp_y = tmp_y.loc[:,KEEP_GROUPS].to_numpy() # reorder
    else:
        pretmp_y = [meta['GROUP'] for meta in tmp_meta if meta['GROUP'] in KEEP_GROUPS]
        tmp_y = np.zeros((tmp_y.shape[0],len(KEEP_GROUPS)))
        for i,c in enumerate(KEEP_GROUPS):
            tmp_y[[j for j,elem in enumerate(pretmp_y) if elem==c],i] = 1
            
    input_data[part]['y'] = tmp_y

# %%

if CORE == "inceptionv3":
    preprocess_input_fn = tf.keras.applications.inception_v3.preprocess_input
    def unpreprocess_input(x, clip=True):
        x += 1.
        x *= 127.5
        if clip:
            x = np.minimum(255, np.maximum(0, x))
        return x.astype('uint')
    unpreprocess_input_fn = unpreprocess_input
elif CORE == "resnet50":
    preprocess_input_fn = tf.keras.applications.resnet50.preprocess_input
    def unpreprocess_input(x, clip=True):
        mean_rgb = [103.939, 116.779, 123.68]
        for c in range(3):
            x[...,c]+=mean_rgb[c]
        # reverse 'RGB'->'BGR', AFTER demeaning
        x = x[..., ::-1]
        if clip:
            x = np.minimum(255, np.maximum(0, x))
        return x.astype('uint')
    unpreprocess_input_fn = unpreprocess_input
elif CORE == 'vgg16':
    preprocess_input_fn = tf.keras.applications.vgg16.preprocess_input
    def unpreprocess_input(x, clip=True):
        mean_rgb = [103.939, 116.779, 123.68]
        for c in range(3):
            x[...,c]+=mean_rgb[c]
        # reverse 'RGB'->'BGR', AFTER demeaning
        x = x[..., ::-1]
        if clip:
            x = np.minimum(255, np.maximum(0, x))
        return x.astype('uint')
    unpreprocess_input_fn = unpreprocess_input
elif CORE == 'efficientnetb4':
    preprocess_input_fn = tf.keras.applications.efficientnet.preprocess_input
    def unpreprocess_input(x, clip=True):
        if clip:
            x = np.minimum(255, np.maximum(0, x))
        return x.astype('uint')
    unpreprocess_input_fn = unpreprocess_input
else:
    assert False, "Unknown backbone: {}".format(CORE)

def getBackbone(core, global_pooling, freeze):
    if core == "inceptionv3":
        core_model = tf.keras.applications.inception_v3.InceptionV3(include_top = False, input_shape = (IM_SIZE, IM_SIZE, 3), pooling = global_pooling)
        
        if freeze == "inputside":
            N_UNFREEZE = 9 # dernier bloc pour Inception v3
            N_CONV = len([l for l in core_model.layers if l.name[:4]=="conv"])
            N_FREEZE = N_CONV-N_UNFREEZE
            n = 1
            for l in core_model.layers:
                if l.name[:4]=="conv" and n<=N_FREEZE:
                    l.trainable = False
                    n += 1
        elif freeze == "outputside":
            N_UNFREEZE = 5 # dernier bloc pour Inception v3
            N_CONV = len([l for l in core_model.layers if l.name[:4]=="conv"])
            N_FREEZE = N_CONV-N_UNFREEZE
            n = 1
            for l in core_model.layers:
                if l.name[:4]=="conv":
                    if n>N_UNFREEZE:
                        l.trainable = False
                    n += 1
        elif freeze!="none":
            assert False, "Unknown freeze: {}".format(freeze)
    
    elif core == "resnet50":
        if freeze == "inputside":
            BLOCKS_FREEZE = ("conv1","conv2","conv3","conv4",) # blocks à freezer
        elif freeze =="outputside":
            BLOCKS_FREEZE = ("conv3","conv4","conv5") # blocks à freezer
        elif freeze=="none":
            BLOCKS_FREEZE = []
        else:
            assert False, "Unknown freeze: {}".format(freeze)

        core_model = tf.keras.applications.resnet50.ResNet50(include_top = False, input_shape = (IM_SIZE, IM_SIZE, 3), pooling = global_pooling)

        FREEZE_LAYERS = [i for i,l in enumerate(core_model.layers) if l.name[:5] in BLOCKS_FREEZE and isinstance(l, tf.keras.layers.Conv2D)]
        for i in FREEZE_LAYERS:
            core_model.layers[i].trainable = False

    elif core == "vgg16":
        if freeze == "inputside":
            BLOCKS_FREEZE = ("block1","block2","block3","block4",) # blocks à freezer
        elif freeze=="outputside":
            BLOCKS_FREEZE = ("block2","block3","block4","block5") # blocks à freezer
        elif freeze=="none":
            BLOCKS_FREEZE = []
        else:
            assert False, "Unknown freeze: {}".format(freeze)
        
        core_model = tf.keras.applications.vgg16.VGG16(include_top = False, input_shape = (IM_SIZE, IM_SIZE, 3), pooling = global_pooling)
        
        if freeze!='none':
            FREEZE_LAYERS = [i for i,l in enumerate(core_model.layers) if l.name[:6] in BLOCKS_FREEZE and l.name[7:11]=="conv"]
            for i in FREEZE_LAYERS:
                core_model.layers[i].trainable = False

    if core == "efficientnetb4":
        core_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top = False, input_shape = (IM_SIZE, IM_SIZE, 3), pooling = global_pooling, weights='imagenet')
        
        if freeze == "inputside":
            BLOCKS_FREEZE = ("block1","block2","block3","block4","block5","block6",) # blocks à freezer
        elif freeze=="outputside":
            BLOCKS_FREEZE = ("block2","block3","block4","block5","block6","block7",) # blocks à freezer
        elif freeze=="none":
            BLOCKS_FREEZE = []
        else:
            assert False, "Unknown freeze: {}".format(freeze)
        if freeze!='none':
            for i,l in enumerate(core_model.layers):
                if (l.name[:6] in BLOCKS_FREEZE) and l.name[-4:]=="conv":
                    # print("Freezing layer: {}".format(l.name))
                    core_model.layers[i].trainable = False
    
    else:
        assert False, "Unknown core: {}".format(core)
                
    return core_model

def getModelWithHyperParameters(core, global_pooling, freeze, nnoutput, base_lr):
    core_model = getBackbone(core, global_pooling, freeze)

    inputs = core_model.input
    
    if nnoutput=="dense":
        x = core_model.output
        output = tf.keras.layers.Dense(N_CLASSES, activation='softmax') (x)
    elif nnoutput=="conv":
        x = core_model.layers[-2].output
        output = tf.keras.layers.Conv2D(N_CLASSES, x.shape[1:-1], activation='softmax') (x)
        output = tf.keras.layers.Reshape(target_shape=(N_CLASSES,)) (output)
        
    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    opt=tf.keras.optimizers.Adam(learning_rate = base_lr)
    
    model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["categorical_accuracy", "AUC"])
    
    return model

if DEBUG:
    model = getModelWithHyperParameters(CORE,"max","inputside","dense",1e-3)
    model = getModelWithHyperParameters(CORE,"avg","outputside","conv",1e-5)
    model = getModelWithHyperParameters("efficientnetb4","avg","none","dense",1e-5)
    
# %%

if FLAGS.step=="train":
    # create generators
    train_gen = ImgAugDataGenerator(images = input_data['train']['images'],
                                    y = input_data['train']['y'],
                                    batch_size = BATCH_SIZE,
                                    width = IM_SIZE,
                                    height = IM_SIZE,
                                    preprocess_function = preprocess_input_fn,
                                    full_batches_only = True,
                                    shuffle=True,
                                    augment=1)
    val_gen = ImgAugDataGenerator(images = input_data['valid']['images'],
                                  y = input_data['valid']['y'],
                                  batch_size = BATCH_SIZE,
                                  width = IM_SIZE,
                                  height = IM_SIZE,
                                  preprocess_function = preprocess_input_fn,
                                  full_batches_only = False,
                                  shuffle=False,
                                  augment=0)
    
    if DEBUG:
        from matplotlib import pyplot as plt
        tmp_x, tmp_y = train_gen.__getitem__(0)
        plt.figure()
        for ix in range(8):
            plt.subplot(3,3,ix+1)
            plt.imshow(tmp_x[ix])
            plt.text(0, 0, KEEP_GROUPS[np.argmax(tmp_y[ix])], color='#ffffff',verticalalignment='top')
        tmp_x, tmp_y = val_gen.__getitem__(0)
        plt.figure()
        for ix in range(8):
            plt.subplot(3,3,ix+1)
            plt.imshow(tmp_x[ix])
            plt.text(0, 0, KEEP_GROUPS[np.argmax(tmp_y[ix])], color='#ffffff',verticalalignment='top')
        
    if FLAGS.training_mode == "kt":
        def getModelHP(hp):
            return getModelWithHyperParameters(CORE,
                                               global_pooling = hp.Choice('global_pooling', values=("avg","max"), default="max"),
                                               freeze = hp.Choice('freeze', values=("inputside","outputside","none"), default="inputside"),
                                               nnoutput = hp.Choice('nnoutput', values=("dense","conv"), default="dense"),
                                               base_lr = hp.Choice('lr', values=(1e-5,5e-5,1e-4,1e-3), default=1e-3))
        
        tuner = kt.tuners.BayesianOptimization(getModelHP,
                                               objective='val_loss',
                                               num_initial_points=3,
                                               max_trials=20,
                                               directory=output_path,
                                               project_name=MODEL_NAME)
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-5, min_delta=1e-3)]
    
        tuner.search(train_gen,
                     epochs = EPOCHS,
                     validation_data = val_gen,
                     callbacks = callbacks,
                     verbose = 2)
        
        best_hyperparams = tuner.get_best_hyperparameters()[0]
        
        hp_df = pd.DataFrame.from_dict(tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters.values, orient='index')
        hp_df.to_csv(os.path.join(output_path, MODEL_NAME+'.csv'))
        
        model = tuner.get_best_models(num_models=1)[0]
        model.save(os.path.join(output_path, MODEL_NAME+'.h5'))
        
    elif FLAGS.training_mode == "man":
        model = getModelWithHyperParameters(CORE,
                                            global_pooling = FLAGS.global_pooling,
                                            freeze = FLAGS.freeze,
                                            nnoutput = FLAGS.nnoutput,
                                            base_lr = FLAGS.base_lr)
        
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_path, MODEL_NAME+'.h5'), monitor='val_loss', save_best_only=True, save_weights_only=False, restore_best_weights=True),
                     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=1e-3, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, min_lr=1e-5, min_delta=1e-3)]
    
        model.fit(train_gen,
                  epochs = EPOCHS,
                  validation_data = val_gen,
                  callbacks = callbacks,
                  verbose = 2)
        
# %%

if FLAGS.step=="test":
    model = tf.keras.models.load_model(os.path.join(output_path, MODEL_NAME+".h5"))
    
    for dset in ('train','valid','test',):
        if DEBUG:
            keep_samples = np.concatenate([np.random.choice(np.where(input_data[dset]['y'][:,0]==1)[0], 8),
                                           np.random.choice(np.where(input_data[dset]['y'][:,1]==1)[0], 8)]).tolist()
            input_data[dset]['images'] = [input_data[dset]['images'][i] for i in keep_samples]
            input_data[dset]['y'] = input_data[dset]['y'][keep_samples,:]
            
        gen = ImgAugDataGenerator(images = input_data[dset]['images'],
                                  y = input_data[dset]['y'],
                                  batch_size = BATCH_SIZE,
                                  width = IM_SIZE,
                                  height = IM_SIZE,
                                  preprocess_function = preprocess_input_fn,
                                  full_batches_only = False,
                                  shuffle=False,
                                  augment=0)
        
        y = input_data[dset]['y']
        y_ = model.predict(gen)
        
        np.save(os.path.join(output_path,'results','{}_{}_gt.npy'.format(MODEL_NAME,dset)), y[:,1])
        np.save(os.path.join(output_path,'results','{}_{}_pred.npy'.format(MODEL_NAME,dset)), y_[:,1])
        
# %%

if FLAGS.step=="test_summary":
    dset='train'
    dset='valid'
    dset='test'
    for dset in ('train','valid','test',):
        # load y and y_
        y = np.load(os.path.join(output_path,'results','{}_{}_gt.npy'.format(MODEL_NAME,dset)))
        y_ = np.load(os.path.join(output_path,'results','{}_{}_pred.npy'.format(MODEL_NAME,dset)))
        
        # get annotations
        y_annot = [os.path.split(os.path.split(f)[0])[-1] for f in input_data[dset]['images']]
        
        import matplotlib.pyplot as plt
        # from misc_ml.viz import plotROC, quickBoxPlot
        
        # plotROC(y,y_)
        # quickBoxPlot(y,y_)
        
        # annot after aggregating
        
        annot_df = pd.DataFrame(dict(Patient=y_annot,Group=y,Pred=y_))
        
        # make bootstraps
        aggreg_results = []
        n_bootstraps = 500
        min_bootstrap_size = 1
        max_bootstrap_size = 50
        from tqdm import tqdm
        from sklearn.metrics import roc_auc_score
        for bootstrap_size in tqdm(range(min_bootstrap_size,max_bootstrap_size+1)):
            for patient in annot_df.Patient.unique():
                bootstrap_preds = np.random.choice(annot_df.Pred[annot_df.Patient==patient], bootstrap_size)
                aggreg_results.append(dict(Truth=annot_df.Group[annot_df.Patient==patient].iloc[0],
                                           NumPreds=bootstrap_size,
                                           MeanPred=np.mean(bootstrap_preds),
                                           MinPred=np.min(bootstrap_preds),
                                           MaxPred=np.max(bootstrap_preds),
                                           MedianPred=np.median(bootstrap_preds)))
        aggreg_results = pd.DataFrame(aggreg_results)
        # plot aggregated results
        auc_results = []
        for bootstrap_size in range(min_bootstrap_size,max_bootstrap_size+1):
            auc_results.append(dict(NumPreds=bootstrap_size,
                                    MeanPredAUC=roc_auc_score(y_true = aggreg_results.Truth.loc[aggreg_results.NumPreds==bootstrap_size], y_score = aggreg_results.MeanPred.loc[aggreg_results.NumPreds==bootstrap_size]),
                                    MinPredAUC=roc_auc_score(y_true = aggreg_results.Truth.loc[aggreg_results.NumPreds==bootstrap_size], y_score = aggreg_results.MinPred.loc[aggreg_results.NumPreds==bootstrap_size]),
                                    MaxPredAUC=roc_auc_score(y_true = aggreg_results.Truth.loc[aggreg_results.NumPreds==bootstrap_size], y_score = aggreg_results.MaxPred.loc[aggreg_results.NumPreds==bootstrap_size]),
                                    MedianPredAUC=roc_auc_score(y_true = aggreg_results.Truth.loc[aggreg_results.NumPreds==bootstrap_size], y_score = aggreg_results.MedianPred.loc[aggreg_results.NumPreds==bootstrap_size])))
        auc_results = pd.DataFrame(auc_results)
        # from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(auc_results.NumPreds, auc_results.MeanPredAUC, label="Mean")
        # plt.plot(auc_results.NumPreds, auc_results.MinPredAUC, label="Min")
        # plt.plot(auc_results.NumPreds, auc_results.MaxPredAUC, label="Max")
        plt.plot(auc_results.NumPreds, auc_results.MedianPredAUC, label="Median")
        plt.xticks(np.arange(min_bootstrap_size,max_bootstrap_size+1))
        plt.legend()
        plt.grid()
        
        # PART BY NAME
        # VGG16 test set : AUC = 0.81 without even need for aggregating, 0.90-0.95 with >=12 images (bootstrap)
        # IncV3 test set : AUC = 0.80 without even need for aggregating, 0.90-0.95 with >=12 images (bootstrap)
        # EffB4 test set : AUC = 0.82 without even need for aggregating, 0.95-1.00 with >=6 images (bootstrap)
        
# %%

if FLAGS.step=="viz":
    model = tf.keras.models.load_model(os.path.join(output_path, MODEL_NAME+".h5"))
    
    dset = 'test'
    
    y = np.load(os.path.join(output_path,'results','{}_{}_gt.npy'.format(MODEL_NAME,dset)))
    y_ = np.load(os.path.join(output_path,'results','{}_{}_pred.npy'.format(MODEL_NAME,dset)))
        
    GRADIENTS_FROM_LAYER = None
    # GRADIENTS_FROM_LAYER = 'conv2d_28'
    if CORE=="inceptionv3":
        GRADIENTS_FROM_LAYER = 'mixed10'
    elif CORE=="efficientnetb4":
        # GRADIENTS_FROM_LAYER = 'top_bn'
        GRADIENTS_FROM_LAYER = 'block7b_project_conv'

    # create our grad cam class
    # based on https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras.models import Model
    import cv2
    class GradCAM:
        def __init__(self, model, classIdx, layerName=None):
            # store the model, the class index used to measure the class
            # activation map, and the layer to be used when visualizing
            # the class activation map
            self.model = model
            self.classIdx = classIdx
            self.layerName = layerName
            
            # if the layer name is None, attempt to automatically find
            # the target output layer
            if self.layerName is None:
                self.layerName = self.find_target_layer()

        def find_target_layer(self):
            # attempt to find the final convolutional layer in the network
            # by looping over the layers of the network in reverse order
            for layer in reversed(self.model.layers):
                # check to see if the layer has a 4D output
                if len(layer.output_shape) == 4:
                    return layer.name
            # otherwise, we could not find a 4D layer so the GradCAM
            # algorithm cannot be applied
            raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
        def compute_heatmap(self, image, scale = False, eps=1e-8):
            # construct our gradient model by supplying (1) the inputs
            # to our pre-trained model, (2) the output of the (presumably)
            # final 4D layer in the network, and (3) the output of the
            # softmax activations from the model
            #gradModel = Model(inputs=[model.inputs],
            #            outputs=[model.get_layer(layerName).output,
            #                     model.output])
            gradModel = Model(inputs=[self.model.inputs],
                        outputs=[self.model.get_layer(self.layerName).output,
                                  self.model.output])
            
            # record operations for automatic differentiation
            with tf.GradientTape() as tape:
                # cast the image tensor to a float-32 data type, pass the
                # image through the gradient model, and grab the loss
                # associated with the specific class index
                inputs = tf.cast(image, tf.float32)
                (convOutputs, predictions) = gradModel(inputs)
                #loss = predictions[:, classIdx]
                loss = predictions[:, self.classIdx]
            # use automatic differentiation to compute the gradients
            grads = tape.gradient(loss, convOutputs)
            
            # compute the guided gradients
            castConvOutputs = tf.cast(convOutputs > 0, "float32")
            castGrads = tf.cast(grads > 0, "float32")
            guidedGrads = castConvOutputs * castGrads * grads
            # the convolution and guided gradients have a batch dimension
            # (which we don't need) so let's grab the volume itself and
            # discard the batch
            convOutputs = convOutputs[0]
            guidedGrads = guidedGrads[0]
            
            # compute the average of the gradient values, and using them
            # as weights, compute the ponderation of the filters with
            # respect to the weights
            weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
            
            # grab the spatial dimensions of the input image and resize
            # the output class activation map to match the input image
            # dimensions
            (w, h) = (image.shape[2], image.shape[1])
            heatmap = cv2.resize(cam.numpy(), (w, h))
            # normalize the heatmap such that all values lie in the range
            # [0, 1], scale the resulting values to the range [0, 255],
            # and then convert to an unsigned 8-bit integer
            numer = heatmap - np.min(heatmap)
            denom = (heatmap.max() - heatmap.min()) + eps
            heatmap = numer / denom
            if scale:
                heatmap = heatmap/np.max(heatmap)
            heatmap = (heatmap * 255).astype("uint8")
            # return the resulting heatmap to the calling function
            return heatmap
        
        def overlay_heatmap(self, heatmap, image, raw_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
            # apply the supplied color map to the heatmap and then
            # overlay the heatmap on the input image
            heatmap_ready = cv2.applyColorMap(255-heatmap, colormap)
            # image_ready = ((image[0]/2.+.5)*255).astype('uint8')
            image_ready = raw_image[0].astype('uint8')
            output_ready = cv2.addWeighted(image_ready, alpha, heatmap_ready, 1 - alpha, 0)
            # return a 2-tuple of the color mapped heatmap and the output,
            # overlaid image
            return (image_ready, heatmap_ready, output_ready)
        
    N_IMGS = 3
        
    # image_indices = (0,1,2,3)
    image_indices = np.concatenate([np.random.choice(np.where( (y==1) & (y_>.99) )[0], N_IMGS),
                                    np.random.choice(np.where( (y==0) & (y_<.01) )[0], N_IMGS)])
    
    for i in image_indices:
        # raw_image = np.array(Image.open(input_data[dset]['images'][i]))
        raw_image = adjust([np.array(Image.open(input_data[dset]['images'][i]))], IM_SIZE, IM_SIZE)[0]
        image = np.expand_dims(preprocess_input_fn(raw_image), 0)
        ground_truth = y[i]
        prediction = y_[i]
        prediction_binary = int(np.round(prediction))
        
        # ground_truth_class = KEEP_GROUPS[ground_truth]
        # predicted_class = KEEP_GROUPS[prediction]
        
        cam = GradCAM(model, prediction_binary, layerName=GRADIENTS_FROM_LAYER)
        heatmap = cam.compute_heatmap(image, scale=False)
        # resize the resulting heatmap to the original input image dimensions
        # and then overlay heatmap on top of the image
        heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
        
        plot_data = list(map(lambda tmpl: dict(image_ready=tmpl[0],heatmap_ready=tmpl[1],output_ready=tmpl[2]), [cam.overlay_heatmap(heatmap, image, np.expand_dims(raw_image, 0), alpha=0.5) for cam,heatmap,image,raw_image in zip([cam],[heatmap],[image],[raw_image])]))
        
        # if EXPORT:
        #     plt.figure(figsize=(len(plot_data)*2,4))
        #     for j,dat in enumerate(plot_data):
        #         plt.subplot(2, len(plot_data), j+1)
        #         plt.imshow(dat["image_ready"])
        #         plt.axis("off")
        #         if AUG_LABEL_TOPRIGHT and j%2!=0:
        #             plt.text(358,2,"Ground Truth: {}".format(EXPORT_CLASSES[CLASS_NAMES[ground_truth_class]]),
        #                       verticalalignment = "top", horizontalalignment = "right", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
        #                       bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
        #         else:
        #             plt.text(2,2,"Ground Truth: {}".format(EXPORT_CLASSES[CLASS_NAMES[ground_truth_class]]),
        #                       verticalalignment = "top", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
        #                       bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
        #         plt.subplot(2, len(plot_data), j+3)
        #         plt.imshow(dat["output_ready"])
        #         plt.axis("off")
        #         if AUG_LABEL_TOPRIGHT and j%2!=0:
        #             plt.text(358,2,"{} ({:.1f}%)".format(EXPORT_CLASSES[CLASS_NAMES[predicted_classes[j]]],100*predictions[j][0,predicted_classes[j]]),
        #                       verticalalignment = "top", horizontalalignment = "right", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
        #                       bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
        #         else:
        #             plt.text(2,2,"{} ({:.1f}%)".format(EXPORT_CLASSES[CLASS_NAMES[predicted_classes[j]]],100*predictions[j][0,predicted_classes[j]]),
        #                       verticalalignment = "top", color="#ffffff", fontfamily="Arial", fontsize=10, fontweight="bold",
        #                       bbox = dict(boxstyle='square', facecolor='black', alpha=1.0))
        #     plt.tight_layout()
        #     plt.show()
        #     figname = "{}_{}.png".format(CLASS_NAMES[ground_truth_class],i)
        #     plt.savefig(os.path.join(tmp_path_out,figname))
        # else:
        plt.figure(figsize=(8,len(plot_data)*3.5))
        for j,dat in enumerate(plot_data):
            plt.subplot(len(plot_data), 3, j*3+1)
            plt.imshow(dat["image_ready"])
            plt.title("True class: {} ({})".format(KEEP_GROUPS[ground_truth],i))
            plt.subplot(len(plot_data), 3, j*3+2)
            plt.imshow(dat["heatmap_ready"])
            plt.title("Heatmap for class {}".format(KEEP_GROUPS[prediction_binary]))
            plt.subplot(len(plot_data), 3, j*3+3)
            plt.imshow(dat["output_ready"])
            plt.title("Confidence: {:.1f}%".format(100*prediction))
        plt.tight_layout()
        plt.show()
    
                













