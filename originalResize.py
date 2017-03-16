def resize(image, X, Y, random = False, size = (50, 50)):
    originalImage = image
    # resize imgage to determined size maintaing the original ratio
    (yMaxBound, xMaxBound, _) = image.shape

    newX = [x/float(xMaxBound) for x in X]
    newY = [y/float(yMaxBound) for y in Y]


    if random:
        ratio = np.random.uniform(0.5, 1)
        size = (int(xMaxBound*ratio), int(yMaxBound*ratio))

    image = Image.fromarray(np.uint8(image))
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    (newXMaxBound, newYMaxBound) = image.size

    newX = [x*float(newXMaxBound) for x in newX]
    newY = [y*float(newYMaxBound) for y in newY]

    thumb = image.crop( (0, 0, size[0], size[1]) )
    image = np.asarray(thumb)

    offset_y = (size[0] - image_size[1]) / 2 
    offset_x = (size[1] - image_size[0]) / 2

    newX = [x + offset_x for x in newX]
    newY = [y + offset_y for y in newY]
    
    thumb = ImageChops.offset(thumb, offset_x, offset_y)



    image = np.asarray(thumb)