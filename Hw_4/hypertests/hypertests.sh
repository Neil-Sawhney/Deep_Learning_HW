./main.py ./runners/classify_cifar10.py -c ./hypertests/1.yaml
./main.py ./runners/classify_cifar10.py -c ./hypertests/2.yaml
./main.py ./runners/classify_cifar10.py -c ./hypertests/3.yaml
./main.py ./runners/classify_cifar10.py -c ./hypertests/4.yaml
./main.py ./runners/classify_cifar10.py -c ./hypertests/5.yaml
git add -A
git commit -m "hypertests"
git push
sudo shutdown -h now