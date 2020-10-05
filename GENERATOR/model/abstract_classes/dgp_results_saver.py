import numpy as np
import os
from os import path 
import json
from abc import ABC, abstractmethod # for abstract methods
from datetime import datetime

class DGP_Results_Saver(ABC):

    """ An abstract class that provide different saving methods """
    
    def __init__(self, dir_path):
        if not path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except OSError:
                print ("Creation of the directory %s failed" % dir_path)
        self.dir_path = dir_path


    @abstractmethod
    def save_all(self, params=None, result=None):
        """
        :param params: DGP_Params instance that will be saved
        :param result: DGP_Tesults instance that will be saved
        """
        pass

    @abstractmethod
    def save_dataframe(self, dataframe, filename):
        """
        :param dataframe: pandas.Dataframe that will be saved
        :param filename: filename where file will be saved
        """
        pass