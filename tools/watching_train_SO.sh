var=0
while [ $var -eq 0 ]
do
	count=0
	for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
	do
		if [ $i -lt 100 ]
		then
			echo 'GPU'$count' is avaiable'
		    CUDA_VISIBLE_DEVICES=$count python trainval_net.py 
			var=1
			break
		fi
		count=$(($count+1))
	done
done