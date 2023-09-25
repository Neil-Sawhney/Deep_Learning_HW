echo "#### CREATING TAR FILE ####"
tar cvf sawhney-neil-2023-hw4.tar ./artifacts ./configs ./data ./modules ./helpers ./runners ./README.md ./requirements.txt ./tests ./data --exclude=./temp/checkpoints
echo "#### TAR FILE CREATED ####"
