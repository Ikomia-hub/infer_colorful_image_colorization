from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class ColorfulImageColorization(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from ColorfulImageColorization.ColorfulImageColorization_process import ColorfulImageColorizationProcessFactory
        # Instantiate process object
        return ColorfulImageColorizationProcessFactory()

    def getWidgetFactory(self):
        from ColorfulImageColorization.ColorfulImageColorization_widget import ColorfulImageColorizationWidgetFactory
        # Instantiate associated widget object
        return ColorfulImageColorizationWidgetFactory()
