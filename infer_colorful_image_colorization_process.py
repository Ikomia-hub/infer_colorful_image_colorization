from ikomia import utils, core, dataprocess
import copy
import numpy as np
import cv2
import os
import requests

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

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        pass

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
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
            self.set_param_object(ColorfulImageColorizationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get input :
        input_img = self.get_input(0)
        src_img = input_img.get_image()
        width_in = 224
        height_in = 224

        # Get parameters :
        param = self.get_param_object()

        # Load the model from disk
        if self.net is None or param.update == True:
            prototxt_path = os.path.dirname(os.path.realpath(__file__)) + "/model/colorizationV2.prototxt"
            model_path = os.path.dirname(os.path.realpath(__file__)) + "/model/colorizationV2.caffemodel"

            if not os.path.exists(model_path):
                print("Downloading model, please wait...")
                model_url = utils.get_model_hub_url() + "/" + self.name + "/colorizationV2.caffemodel"
                # self.download(model_url, model_path)
                response = requests.get(model_url, stream=True)
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)


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
        self.emit_step_progress()

        # Set network input
        self.net.setInput(cv2.dnn.blobFromImage(img_l_rs))
        # Get the result
        ab_dec = self.net.forward()[0,:,:,:].transpose((1,2,0))
        ab_dec_us = cv2.resize(ab_dec, (width, height))

        # Step progress bar:
        self.emit_step_progress()

        # Concatenate with original image L
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_dec_us), axis=2)
        img_rgb_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)

        # Get task output :
        output_img = self.get_output(0)

        # Set image of output (numpy array):
        img_rgb_out = img_rgb_out * 255
        img_rgb_out = img_rgb_out.astype(np.uint8)

        output_img.set_image(img_rgb_out)
      
        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

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
        self.info.short_description = "Automatic colorization of grayscale image based on neural network."
        self.info.authors = "Richard Zhang, Phillip Isola, Alexei A. Efros"
        self.info.article = "Colorful Image Colorization"
        self.info.journal = "ECCV"
        self.info.year = 2016
        self.info.documentation_link = "https://richzhang.github.io/colorization/"
        self.info.license = "BSD 2-Clause 'Simplified' License"
        self.info.repository = "https://github.com/Ikomia-hub/infer_colorful_image_colorization"
        self.info.original_repository = "https://github.com/richzhang/colorization"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Colorization"
        self.info.icon_path = "icon/icon.png"
        self.info.keywords = "deep,learning,caffe,CNN,photo"
        self.info.version = "1.2.0"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "COLORIZATION"

    def create(self, param=None):
        # Create process object
        return ColorfulImageColorization(self.info.name, param)
