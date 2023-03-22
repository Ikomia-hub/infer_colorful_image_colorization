from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        from infer_colorful_image_colorization.infer_colorful_image_colorization_process import ColorfulImageColorizationFactory
        # Instantiate process object
        return ColorfulImageColorizationFactory()

    def get_widget_factory(self):
        from infer_colorful_image_colorization.infer_colorful_image_colorization_widget import ColorfulImageColorizationWidgetFactory
        # Instantiate associated widget object
        return ColorfulImageColorizationWidgetFactory()
