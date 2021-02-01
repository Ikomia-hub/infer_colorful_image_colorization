from ikomia import dataprocess
import ColorfulImageColorization_process as processMod
import ColorfulImageColorization_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class ColorfulImageColorization(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.ColorfulImageColorizationProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.ColorfulImageColorizationWidgetFactory()
