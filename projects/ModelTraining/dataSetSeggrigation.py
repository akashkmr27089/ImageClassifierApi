

import splitfolders

inputDir = "D:\Research\CarControl\CarApi\projects\ModelTraining\dataset\input"
outputDir = "D:\Research\CarControl\CarApi\projects\ModelTraining\dataset\Output"

#Take files and split it into validation, training , testing folder
splitfolders.ratio(inputDir, output=outputDir, seed=1337, ratio=(.5, 0.25,0.25)) 