python ../src/main.py \
    --model ResNet101 \
    --residual add

#python ../src/main.py \
#    --model ResNet18 \
#    --residual concat

python ../src/main.py \
    --model ResNet101 \
    --residual minus

python ../src/main.py \
    --model ResNet101 \
    --residual mul

python ../src/main.py \
    --model ResNet101 \
    --residual identity
