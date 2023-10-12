./main.py ./runners/classify_cifar10.py -c ./hypertests/1.yaml
git add -A
git commit -m "hypertests: cifar10"
git push

./main.py ./runners/classify_cifar100.py -c ./hypertests/1.yaml
git add -A
git commit -m "hypertests: cifar100"
git push

runpodctl stop pod rj0q47mezg2nw7