# RT Deep RRT Star
 State of the Art RRT Star Implementation using Deep Neural Networks. 

## Step 1 : Dataset Management ( Till Block 2)

Creates Dataset including 200 start, goal points for each environment and stores it in the csv file in a format Comaptible with the Data Loader.

## Step 2: RRT Star Data Creator

This piece of code writes the 200 paths for all the 200 environment in output_final.csv file.


## Step 3 : Dataset Management (From Block 3 till Final Block)

After the ouput_final.csv id created there might be instances where the dataset has to be put into the right format. These blocks of code are meant to correct the oytput_final.csv file.

## Step 4: CAE_training

This code includes Contractive AutoEncoder which can be run to obtain encoder weights that will help us encode any given environment from 128 x 128 to 28 x 1. 

## Step 5: Neural Planner Trainer

After the dataset is created in the output_final.csv it can be uploaded in loaded in the Data Loader and the MLP Model can be trained to learn the behavior of RRT-Star Implementation. The model weights obtained are stored in  a ".pkl" file.

## Step 6: Static_env_neural_planner

After training the MLP Model. Any test dataset image with start and goal positions can be specified and the code will provide the path based on the trained model weights.

## Step 7: Path Visualisation

If you want to visualize the obtained path.

## Step 8: Dynamic_env_neural_planner

This code is designed to implement Deep RRT Star in Dynamic Environments. If there is any change in the environment. The neural planner takes the new sample space in consideration and provides the next steps accordingly.

## Datasets

You can find the Datasets for the Repository [here](https://drive.google.com/drive/folders/1WhTzoJnMiI-XtYcoDwNPRzP7mPWXjQmy?usp=sharing)