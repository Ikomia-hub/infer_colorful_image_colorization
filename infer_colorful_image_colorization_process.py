from ikomia import utils, core, dataprocess
import copy
import numpy as np
import cv2
import os


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class ColorfulImageColorizationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class ColorfulImageColorization(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.net = None
        self.pts_in_hull = np.load(os.path.dirname(os.path.realpath(__file__)) + "/model/pts_in_hull.npy")
        self.pts_in_hull = self.pts_in_hull.transpose().reshape(2, 313, 1, 1)

        # Create parameters class
        if param is None:
            self.setParam(ColorfulImageColorizationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        input_img = self.getInput(0)
        src_img = input_img.getImage()
        width_in = 224
        height_in = 224

        # Get parameters :
        param = self.getParam()

        # Load the model from disk
        if self.net is None or param.update == True:
            prototxt_path = os.path.dirname(os.path.realpath(__file__)) + "/model/colorizationV2.prototxt"
            model_path = os.path.dirname(os.path.realpath(__file__)) + "/model/colorizationV2.caffemodel"

            if not os.path.exists(model_path):
                print("Downloading model, please wait...")
                model_url = utils.getModelHubUrl() + "/" + self.name + "/colorizationV2.caffemodel"
                self.download(model_url, model_path)

            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.setup_colorization_layer()
            self.net.setPreferableBackend(param.backend)
            self.net.setPreferableTarget(param.target)
            param.update = False

        # Call to the process main routine
        if src_img.ndim == 2:
            img_rgb = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = src_img

        img_rgb = (img_rgb * 1.0 / 255).astype(np.float32)

        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        # Pull out L channel
        img_l = img_lab[:, :, 0]
        # Original image size
        (height, width) = img_rgb.shape[:2]

        # Resize image to network input size
        img_rs = cv2.resize(img_rgb, (width_in, height_in))
        img_lab_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2Lab)
        img_l_rs = img_lab_rs[:, :, 0]
        # Subtract 50 for mean-centering
        img_l_rs -= 50

        # Step progress bar:
        self.emitStepProgress()

        # Set network input
        self.net.setInput(cv2.dnn.blobFromImage(img_l_rs))
        # Get the result
        ab_dec = self.net.forward()[0,:,:,:].transpose((1,2,0))
        ab_dec_us = cv2.resize(ab_dec, (width, height))

        # Step progress bar:
        self.emitStepProgress()

        # Concatenate with original image L
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
        img_rgb_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)

        # Get task output :
        output_img = self.getOutput(0)

        # Set image of output (numpy array):
        output_img.setImage(img_rgb_out)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def setup_colorization_layer(self):
        self.net.getLayer(self.net.getLayerId('class8_ab')).blobs = [self.pts_in_hull.astype(np.float32)]
        self.net.getLayer(self.net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class ColorfulImageColorizationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_colorful_image_colorization"
        self.info.shortDescription = "Automatic colorization of grayscale image based on neural network."
        self.info.description = "Given a grayscale photograph as input, " \
                                "this paper attacks the problem of hallucinating " \
                                "a plausible color version of the photograph. " \
                                "This problem is clearly underconstrained, so previous approaches have " \
                                "either relied on significant user interaction or resulted in desaturated colorizations. " \
                                "We propose a fully automatic approach that produces vibrant and realistic colorizations. " \
                                "We embrace the underlying uncertainty of the problem by posing it as a classification task " \
                                "and use class-rebalancing at training time to increase the diversity of colors in the result. " \
                                "The system is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images. " \
                                "We evaluate our algorithm using a ???colorization Turing test,??? asking human participants to choose between a generated and ground truth color image. " \
                                "Our method successfully fools humans on 32 % of the trials, significantly higher than previous methods. " \
                                "Moreover, we show that colorization can be a powerful pretext task for self-supervised feature learning, " \
                                "acting as a cross-channel encoder. This approach results in state-of-the-art performance on several feature learning benchmarks."
        self.info.authors = "Richard Zhang, Phillip Isola, Alexei A. Efros"
        self.info.article = "Colorful Image Colorization"
        self.info.journal = "ECCV"
        self.info.year = 2016
        self.info.documentationLink = "https://richzhang.github.io/colorization/"
        self.info.license = "BSD 2-Clause 'Simplified' License"
        self.info.repository = "https://github.com/richzhang/colorization"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Colorization"
        self.info.iconPath = "icon/icon.png"
        self.info.keywords = "deep,learning,caffe,CNN,photo"
        self.info.version = "1.1.0"

    def create(self, param=None):
        # Create process object
        return ColorfulImageColorization(self.info.name, param)
