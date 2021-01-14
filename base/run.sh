# END=5
# for i in $(seq 1 $END)
# do 
# 	echo $i
# 	var="${i}.py"
# 	echo $var
# 	echo ${i}.py
# done

cp ./bk/* ./
# rm -r ./saved_models/*
# declare -a arr=("1.py" "2.py" "3.py" "4.py" "5.py")
# for i in "${arr[@]}"
END=4
for i in $(seq 4  $END)
do
	rm config.py
	mv ./${i}.py config.py
	python3 main.py

done
