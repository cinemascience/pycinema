outputs = []

#
# add some image metadata to the input images
#
# In this example, we are creating a new column called 'opacity' based
# on the value in the 'phi' column of the existing data
#

# iterate over images 
for image in inputs: 

    # make a new image
    newimage = image.copy()

    # create a new metadata value, based on existing metadata 
    newcol = "opacity"
    if newcol not in image.meta:
        if int(newimage.meta['phi']) < 0: 
            newimage.meta[newcol] = 1.0
        else:
            newimage.meta[newcol] = 0.25
    else:
        print(image.meta)
        print("ERROR: '" + newcol + "' already exists")

    # add to outputs
    outputs.append( newimage ) 
