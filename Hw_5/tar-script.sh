echo "#### CREATING TAR FILE ####"
tar cvf sawhney-neil-2023-hw5.tar --exclude=./temp/checkpoints --exclude=./data --exclude=**/*/.pytest_cache --exclude=**/*/__pycache__ --exclude=./artifacts/**/model ./setup.sh ./artifacts ./train.py ./test.py ./configs ./modules ./helpers ./runners ./README.md ./requirements.txt ./tests
echo "#### TAR FILE CREATED ####"
