from features import *
from reduceData100 import *
from runClassifiers import *

def main() :


	globalIndex=0
	globalAccuracy=0.0
	coefReduction=1

	globalAccuracy=featureSelection(globalIndex,coefReduction)
	print(globalAccuracy)
	size,sizereduced=reduceDataset(globalIndex)
	globalAccuracy=featureSelection(globalIndex+1,coefReduction)
	print(globalAccuracy)
	
	return


if __name__ == "__main__" :
	sys.exit( main() )
