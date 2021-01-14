cat /usr/local/cuda/version.txt
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
python3 -c 'import torch; print(torch.__version__)'
python3 -c 'import sys; print (sys.version)'


