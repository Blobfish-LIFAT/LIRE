class Config:
    __instance = None
    device_ = None

    @staticmethod
    def device():
        return Config.getInstance().device_

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Config.__instance is None:
            Config()
        return Config.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Config.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Config.__instance = self
