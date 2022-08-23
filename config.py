import jstyleson
from bunch import bunchify
from munch import DefaultMunch
import json
from types import SimpleNamespace


class FromJson:
    def __init__(self):
        pass

    def load_from_json(self, json):
        data = jstyleson.loads(json)
        for key, value in self.__dict__.items():
            setattr(self, key, data[key])


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class ConfigLoader:
    def __init__(self):
        self.data = None

    def load_json(self, config_json_path):
        self.data = jstyleson.load(open(config_json_path), object)
        return DefaultMunch.fromDict(self.data)


class Config:
    pass


class Graph(FromJson):
    def __init__(self):
        super().__init__()
        self.adj_matrix = []
        self.graph_sigma = 0


class Alpha_Function(FromJson):
    def __init__(self, alpha_bar=0, sigma=0, speed=0):
        super().__init__()
        self.alpha_bar = alpha_bar,
        self.sigma = sigma
        self.speed = speed


config = jstyleson.load(open('config.json'))