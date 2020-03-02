rm ./log/model_best.pth.tar
# TODO: create shell script for running the testing code of the baseline model
wget https://www.dropbox.com/s/qixv10r8i45n73l/model_best.pth.tar?dl=1
mv model_best.pth.tar?dl=1 ./log/model_best.pth.tar
RESUME='./log/model_best.pth.tar'
python3 test_handin.py --resume $RESUME --data_dir $1 --save_dir $2

