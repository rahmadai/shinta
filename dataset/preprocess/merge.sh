for i in `seq 1 32`;
do
   cat output_${i}.txt >> merged.txt
done
