./main.py ./runners/classify_cifar10.py -c ./hypertests/1.yaml
git add -A
git commit -m "hypertests: 1"
git push

./main.py ./runners/classify_cifar10.py -c ./hypertests/2.yaml
git add -A
git commit -m "hypertests: 2"
git push

./main.py ./runners/classify_cifar10.py -c ./hypertests/3.yaml
git add -A
git commit -m "hypertests: 3"
git push

./main.py ./runners/classify_cifar10.py -c ./hypertests/4.yaml
git add -A
git commit -m "hypertests: 4"
git push

./main.py ./runners/classify_cifar10.py -c ./hypertests/5.yaml
git add -A
git commit -m "hypertests: 5"
git push

sudo shutdown -h now