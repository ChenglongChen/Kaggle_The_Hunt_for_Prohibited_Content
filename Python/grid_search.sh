#!/bin/bash
#####################################################################
# Code provided here for training VW for                            # 
# Kaggle's The Hunt for Prohibited Content Competition              #
# http://www.kaggle.com/c/avito-prohibited-content                  #
#                                                                   #
# It contains mainly 2 stages                                       #
# stage 1: grid search for the best params                          #
# stage 2: bagging to stabilize the results                         #
#                                                                   #
# You should first run the following to generate VW training data:  #
# python generate_vw_file.py                                        #
#                                                                   #
# Version: 1.0 at Sep 01 2014                                       #
# Author: Chenglong Chen < yr@Kaggle >                              #
# Email: c.chenglong@gmail.com                                      #
#####################################################################


###########
## Setup ##
###########
# shell
SHELL=/bin/sh
# VW
VW=~/Software/vowpal_wabbit/vowpalwabbit/vw
# path-to-data & model & log & submission
DATADIR=../Data
TMPDIR=../Tmp
SUBDIR=../Submission
sudo mkdir -p ${TMPDIR}
sudo mkdir -p ${SUBDIR}

# training and testing data
# extract the model arg
model=$1
if [ $model = mixed ]; then
	VWTRAIN=${DATADIR}/train.vw
else if [ $model = proved ]; then
	VWTRAIN=${DATADIR}/train_proved.vw
else if [ $model = unproved ]; then
	VWTRAIN=${DATADIR}/train_unproved.vw
else
	echo "Wrong type of model"
fi; fi; fi
VWTRAINTMP=${DATADIR}/train_tmp.vw
VWTEST=${DATADIR}/test.vw


#############################
## VW params configuration ##
#############################
# general params configuration for VW
# n-gram
NGRAM=2
# l1 regularization
L1=0.00000
# l2 regularization
L2=0.00000
# number of bits for weights
BIT=28
# random seed
SEED=1234
NUMTRAIN=3500000
# training configuration for VW
PARAMS="-q TC -q TS -q Te -q Tp -q Tu -q SP -q Se -q Sp -q Su -q AP -q Ci --ignore j \
--ngram ${NGRAM} --skips 0 -b ${BIT} -c -k --holdout_off --random_seed ${SEED}"

# grid search for hyper-params
LOSS=(logistic)
LR=(0.5 0.2)
LRDECAY=(0.5 0.75 1.0)
PASSES=(2 4 6 8 10 12 14 16)
L1=(0 1e-8)
L2=(0 1e-8)
WEIGHT=(1)

# if you want to save time just use the following params I found using grid search
if [ $model = mixed ]; then
	LR=(0.5)
	LRDECAY=(1.0)
	PASSES=(10)
else if [ $model = proved ]; then
	LR=(0.2)
	LRDECAY=(1.0)
	PASSES=(16)
fi; fi
LOSS=(logistic)
L1=(0)
L2=(0)
WEIGHT=(1)

# number of rounds for bagging (to stabilize the results)
iteration=50


##############
## Training ##
##############
#: << "END"
for loss in ${LOSS[@]}; do
	# create folder
	SUBFOLDER=${SUBDIR}/${model}_${loss}
	#echo "Create folder ${SUBFOLDER}"
	sudo mkdir -p ${SUBFOLDER}
	touch ${SUBFOLDER}/vw.log
	# find the best params with best AP@k on validation set
	BESTVALIDAPatK=0.0
	BESTLR=0.5
	BESTLRDECAY=1.0
	BESTPASSES=5
	BESTL1=0.0
	BESTL2=0.0
	BESTWEIGHT=1

	#############
	## Stage 1 ##
	#############
	for weight in ${WEIGHT[@]}; do
		python generate_weighted_sample.py ${VWTRAIN} ${VWTRAINTMP} 1 ${weight}
		for l1 in ${L1[@]}; do
			for l2 in ${L2[@]}; do
				for lr in ${LR[@]};	do
					for lr_decay in ${LRDECAY[@]}; do
						# --save_resume in VW doesn't seem to work...
						# so I use this computionally espensive method
						for passes in ${PASSES[@]}; do
						
							VALIDPARAMS="$PARAMS --loss_function $loss --learning_rate $lr \
										--decay_learning_rate $lr_decay --passes $passes --l1 ${l1} --l2 ${l2}"
							# train VW model with the first NUMTRAIN training data
							head -n ${NUMTRAIN} ${VWTRAINTMP} | ${VW} ${VALIDPARAMS} --quiet -f ${TMPDIR}/model
							# compute AP@k on the rest data
							tail -n +${NUMTRAIN} ${VWTRAINTMP} | ${VW} -t --quiet -i ${TMPDIR}/model -r ${TMPDIR}/pred
							python generate_submission.py ${TMPDIR}/pred ${TMPDIR}/sub.csv
							VALIDAPatK=$(python APatK.py ${TMPDIR}/sub.csv)
							echo " * Train for: loss=$loss, lr=$lr, lr_decay=$lr_decay, passes=$passes, l1=$l1, l2=$l2, weight=$weight | AP@k=$VALIDAPatK" | tee -a ${SUBFOLDER}/vw.log
							if [ $(awk -v n1=$BESTVALIDAPatK -v n2=$VALIDAPatK 'BEGIN{print(n1<=n2)?"1":"0"}') -eq 1 ]; then
								BESTVALIDAPatK=$VALIDAPatK
								BESTLR=$lr
								BESTLRDECAY=$lr_decay
								BESTPASSES=$passes
								BESTL1=$l1
								BESTL2=$l2
								BESTWEIGHT=$weight
							fi
						done
					done
				done			
			done
		done
	done
	# delete tmp data
	sudo rm -rf ${VWTRAINTMP}.cache ${VWTRAINTMP}.cache.writing ${TMPDIR}/model ${TMPDIR}/pred ${TMPDIR}/sub.csv

	# extract the best params
	echo " * Best params for loss=$loss:" | tee -a ${SUBFOLDER}/vw.log
	echo "   lr=$BESTLR, lr_decay=$BESTLRDECAY, passes=$BESTPASSES, l1=$BESTL1, l2=$BESTL2, weight=$BESTWEIGHT | AP@k=$BESTVALIDAPatK" | tee -a ${SUBFOLDER}/vw.log
	FINALPARAMS="$PARAMS --loss_function $loss --learning_rate $BESTLR \
   				--decay_learning_rate $BESTLRDECAY --passes $BESTPASSES --l1 ${BESTL1} --l2 ${BESTL2}"

	#############
	## Stage 2 ##
	#############
	# train bagging version with the best params to stabilize the results
	python generate_weighted_sample.py ${VWTRAIN} ${VWTRAINTMP} 1 ${BESTWEIGHT}
	for iter in $(seq 1 $iteration); do
		echo
		echo " * Train for bagging iteration: $iter"
		echo "   - Prepare bootstrap samples"
		TMPDATA=${TMPDIR}/bootstrap_samples
		python generate_bootstrap.py ${VWTRAINTMP} ${TMPDATA} $iter
		# since duplicated bootstrap samples are written one after another
		# we have to shuffle the data
		shuf ${TMPDATA} -o ${TMPDATA}
		${VW} ${FINALPARAMS} --quiet -d ${TMPDATA} -f ${TMPDIR}/model
		${VW} -t --quiet -i ${TMPDIR}/model -d ${VWTEST} -r ${TMPDIR}/pred
		python generate_submission.py ${TMPDIR}/pred ${SUBFOLDER}/$iter.csv
		# delete tmp data
		sudo rm -rf ${TMPDATA} ${TMPDATA}.cache ${TMPDATA}.cache.writing ${TMPDIR}/model ${TMPDIR}/pred
	done
	# change the name of the folder
	SUBFOLDER2=${SUBDIR}/${model}_${loss}_lr${BESTLR}_lrdecay${BESTLRDECAY}_pass${BESTPASSES}
	sudo mv ${SUBFOLDER} ${SUBFOLDER2}
done

# delete tmp data
sudo rm -rf ${TMPDIR} ${VWTRAINTMP}
#END

