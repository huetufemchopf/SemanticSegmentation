# TODO: create shell script for running the testing code of your improved model
bash get_data_hw.sh
RESUME='./log/model_bestbest.pth.tar'
python3 test_handin_improvedmodel.py --resume $RESUME --data_dir $1 --save_dir $2
