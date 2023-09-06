#!/bin/bash
#ssh -X nukresearch5
#bash -l
#SPLITS="1 2 3 4 5 6 7 8 9 10"
#for number_splits in $SPLITS; do
#	echo "gate main_normalized_v82_"$number_splits".mac >> output-"$number_splits".txt &"
#done

path=$1
ncpu=$2
currentPath=$pwd
cd $path

for number_splits in $( eval echo {1..$ncpu}); do
	docker run -i --rm -v $PWD:/APP opengatecollaboration/gate main_normalized_${number_splits}.mac >> output-${number_splits}.txt 2>&1 &
	echo "gate main_normalized_"$number_splits".mac >> output-"$number_splits".txt 2>&1 &"
done

cd $currentPath



