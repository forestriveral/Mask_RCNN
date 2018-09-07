

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='../datasets'
dataType='train'
annFile='%s/annotations/%s.json' % (dataDir,dataType)

coco=COCO(annFile)

print("=================")

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n\n', ' '.join(nms))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n', ' '.join(nms))

catIds = coco.getCatIds(catNms = ['bubble'])
imgIds = coco.getImgIds(catIds = catIds)
# print(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# print(img)

I = io.imread('%s/annotations/%s' % (dataDir, img['file_name']))
plt.figure()
plt.axis('off')
plt.imshow(I)
# plt.show()

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# print(annIds)
anns = coco.loadAnns(annIds)
# print(anns)
coco.showAnns(anns)
plt.show()







