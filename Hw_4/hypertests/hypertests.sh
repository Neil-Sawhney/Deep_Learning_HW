./main.py ./runners/classify_cifar10.py -u ./hypertests/1.yaml
git add -A
git commit -m "hypertests: 1"
git push

./main.py ./runners/classify_cifar10.py -u ./hypertests/2.yaml
git add -A
git commit -m "hypertests: 2"
git push

./main.py ./runners/classify_cifar10.py -u ./hypertests/3.yaml
git add -A
git commit -m "hypertests: 3"
git push

./main.py ./runners/classify_cifar10.py -u ./hypertests/4.yaml
git add -A
git commit -m "hypertests: 4"
git push

shutdown -h now