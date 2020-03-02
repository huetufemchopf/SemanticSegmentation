



#€• Semantic Segmentation

![illustration](seg.png)


This is part of the Projects of the course "Deep Learning and Computer Vision" of National Taiwan University.  
In this project, a segmantic segmentation network is implemented. The final mean-iou score is 73.3 % 
In HW2 problem 1, you will need to implement two semantic segmentation models and answer some questions in the report.

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `hw2_data`. Note that this command by default only works on Linux. If you are using other operating systems or you can not download the dataset by running the command above, you should download the dataset from [this link](https://drive.google.com/file/d/1Lp3KS9Gh1LZx6_WVQsSd5H0iHmFAsmFn/view?usp=sharing) and unzip the compressed file manually.

## Run

Run the following command 
 
In order to evaluate the model, run


    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash hw2.sh $1 $2
    CUDA_VISIBLE_DEVICES=GPU_NUMBER bash hw2_best.sh $1 $2
where `$1` is the testing images directory (e.g. `test/images/`), and `$2` is the output prediction directory (e.g. `test/labelTxt_hbb_pred/` )

### Packages
This work should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

