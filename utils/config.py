import jstyleson
from bunch import bunchify
from munch import DefaultMunch
import json
from types import SimpleNamespace

config = jstyleson.load(open('utils/config.json'))
