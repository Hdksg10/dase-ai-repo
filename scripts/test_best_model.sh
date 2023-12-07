python ../src/main.py \
    --model LeNet \
    --test

python ../src/main.py \
    --model AlexNet \
    --test

python ../src/main.py \
    --model DenseNet \
    --gr 32 \
    --test

python ../src/main.py \
    --model SEResNet18 \
    --ratio 4 \
    --test
    
python ../src/main.py \
    --model ResNet18 \
    --residual add \
    --test

python ../src/main.py \
    --model ResNet34 \
    --residual add \
    --test

python ../src/main.py \
    --model ResNet50 \
    --residual add \
    --test

python ../src/main.py \
    --model ResNet101 \
    --residual add \
    --test