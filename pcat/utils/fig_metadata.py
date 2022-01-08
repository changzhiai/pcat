# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:43:32 2021

@author: changai
"""

from exif import Image

class FigsMetaData:
    """Class for identifying pictures"""
    def __init__(self, fig_names='none', file_name='none', sheet='none', min_col=0, max_col=0, min_row=0, max_row=0):
        self.fig_names = fig_names
        self.file_name = file_name
        self.sheet = sheet
        self.min_col = min_col
        self.max_col = max_col
        self.min_row = min_row
        self.max_row = max_row
        
    def add_metadata(self):
        """Add add metadata to figure"""
        row_of_tag = self.min_row-1
        col_of_tag = self.min_col-1
        images = []
        for i, image in enumerate(self.fig_names):
            with open(image, "rb") as file:
                images.append(Image(file)) #read figures via exif
        print('add metadata: \n', images, '\n')
        for i, image in enumerate(images):
            if image.has_exif:
                status = f"contains EXIF (image_description: {image.image_description}) information."
            else:
                status = "does not contain any EXIF information before loading."
            print(f"Image {i} {status}")
            # add metadata to figures
            images[i].image_description = 'file:'+self.file_name+'; sheet:'+self.sheet  #corresponds to data source
            images[i].Model = 'row_of_tag:'+str(row_of_tag)+'; col_of_tag:'+str(col_of_tag)+'; min_col:'+str(self.min_col)+'; max_col:'+str(self.max_col)+'; min_row:'+str(self.min_row)+'; max_row:'+str(self.max_row)
            images[i].copyright = "dtu: changai"
            print(f"Description: {images[i].image_description}")
            print(f"Data scope: {images[i].Model}")
            print(f"Copyright: {images[i].copyright} \n")
        # rewrite figures
        for i, image in enumerate(self.fig_names):
            with open(image, "wb") as file:
                file.write(images[i].get_file())