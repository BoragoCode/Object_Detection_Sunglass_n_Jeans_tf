# Object_Detection_Sunglass_n_Jeans_tf
Custom object detector - Sunglass and Jeans

Custom training for Sunglasses and Jeans Class:
  1. train and test - contain training and testing data along with bbox information
  2. Set up the Tensorflow Object Detection API - See Tensorflow.
  3. Generate tf records - train.csv and test.csv file are already shared, so dont need to convert the XMLs to CSV
  4. Train model - Mobilenet.
  5. Export the model - This will create .pb file from check point file. 
  6. Use Image and Video demo files, remember to update the model dir path.
