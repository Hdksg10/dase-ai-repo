python ../src/main.py \
    --model ResNet50 \
    --residual add

#python ../src/main.py \
#    --model ResNet18 \
#    --residual concat

python ../src/main.py \
    --model ResNet50 \
    --residual minus

python ../src/main.py \
    --model ResNet50 \
    --residual mul

python ../src/main.py \
    --model ResNet50 \
    --residual identity
