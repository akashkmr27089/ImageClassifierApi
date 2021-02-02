import os 
import pandas as pd 
import EnvConfig

'''
This is used for creating Labels 
    1. EnvConfig.trinDir 
'''

## Making CSV file for Labels 
trainLabels = os.listdir(EnvConfig.trainDir)
trainLabelsArray = []
for i, x in enumerate(trainLabels):
    temp = x.split('.')
    trainLabelsArray.append([temp[1], 1 if temp[0]=='cat' else 0])
trainLabelsPandas = pd.DataFrame(trainLabelsArray, columns = ['id', 'label'])

trainLabelsPandas.to_csv(EnvConfig.RootDir + '/TrainLabels.csv', index = False)


'''
Script to Shifting all the data to 
'''

# import splitfolders

# inputDir = EnvConfig.trainDir
# outputDir = EnvConfig.RootDir + '/dataset/Output'

# #Take files and split it into validation, training , testing folder
# splitfolders.ratio(inputDir, output=outputDir, seed=1337, ratio=(.75, 0.25)) 


'''
Image Transformation to see images 
'''
# plt.imshow(features[0].numpy().transpose(1,2,0))