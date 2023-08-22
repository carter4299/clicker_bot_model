import os
import numpy as np

def list_svgs_in_assets(folder_name):
    all_files = os.listdir(folder_name)
    svg_files = [f for f in all_files if f.endswith('.png')]
    return svg_files


def get_labels(folder_name):
    extra_labels_dict = {}
    pngs = list_svgs_in_assets(folder_name)
    for png in pngs:
        extra_labels_dict[png] = int(png[0])
    
    return extra_labels_dict

def standardize_images(X):
    return (X - (X.mean().astype(np.float32))) / (X.std().astype(np.float32))

"""
# Uncomment this to run this file independently
if __name__ == "__main__":
    #pngs = list_svgs_in_assets(folder_name='./')
    #lables_dict = get_labels()"""