./main.py ./runners/classify_cifar10.py -c ./hypertests/preston.yaml
./main.py ./runners/classify_cifar10.py -c ./hypertests/1.yaml
git add -A
git commit -m "hypertests"
git push
sudo shutdown -h now