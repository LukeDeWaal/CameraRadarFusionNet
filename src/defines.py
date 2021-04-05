import sys, os, getpass

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
LOC_MINI_DIR = os.path.join(ROOT_DIR, 'data', 'mini')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
CONFIG_DIR = os.path.join(ROOT_DIR, 'configs')
EXT_DATA_DIR = os.path.join('/media', getpass.getuser(), '2TB-HDD')
EXT_MINI_DIR = os.path.join(EXT_DATA_DIR, 'mini')
EXT_FULL_DIR = os.path.join(EXT_DATA_DIR, 'full')


