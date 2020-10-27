from PIL import Image

filename = r'E:\data\yangben\0.jpg'
img = Image.open(filename)
imgSize = img.size #图片的长和宽
print (imgSize)