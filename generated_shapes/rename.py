import os
from PIL import Image
import numpy as np

filenames = os.listdir(".")

for filename in filenames:
    file_root, file_ext = os.path.splitext(filename)
    if file_root[-5:] == "_proc":
	new_file_root = file_root[:-5]
	os.rename(filename, new_file_root + ".obj")
    

