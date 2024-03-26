import pandas as pd

from os.path import abspath, dirname, join

import OgreInterface

BASE_PATH = dirname(abspath(OgreInterface.__file__))

ionic_radii_df = pd.read_csv(join(BASE_PATH, "data", "ionic_radii_data.csv"))
