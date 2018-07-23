""" Settings and configuration for yolo

Read values from global_settings.py
see global_settings.py for all possible
variables
"""
import conf.global_settings


class Settings:
    def __init__(self, setting_modules):

        #constructing attributes
        for settings in dir(setting_modules):
            if settings.isupper():
                setattr(self, settings, getattr(global_settings, settings))

settings = Settings(global_settings)


