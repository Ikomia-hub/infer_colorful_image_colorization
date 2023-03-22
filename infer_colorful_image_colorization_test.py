import logging
import cv2
from ikomia.utils.tests import run_for_test


logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::infer colorful image colorization =====")
    logger.info("----- Use default parameters")
    img = cv2.imread(data_dict["images"]["detection"]["coco"], cv2.IMREAD_GRAYSCALE)
    input_img_0 = t.get_input(0)
    input_img_0.set_image(img)
    return run_for_test(t)
