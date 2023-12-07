python ../src/main.py \
    --model ResNet18 \
    --residual add

#python ../src/main.py \
#    --model ResNet18 \
#    --residual concat

python ../src/main.py \
    --model ResNet18 \
    --residual minus

python ../src/main.py \
    --model ResNet18 \
    --residual mul

python ../src/main.py \
    --model ResNet18 \
    --residual identity
