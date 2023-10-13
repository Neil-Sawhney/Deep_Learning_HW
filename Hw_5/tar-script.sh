echo "#### CREATING TAR FILE ####"
tar cvf sawhney-neil-2023-hw4.tar --exclude=./temp/checkpoints --exclude=./data --exclude=**/*/__pycache__ ./artifacts ./train.py ./test.py ./configs ./modules ./helpers ./runners ./README.md ./requirements.txt ./tests
echo "#### TAR FILE CREATED ####"
