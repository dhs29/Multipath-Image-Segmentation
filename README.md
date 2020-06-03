AIM : Multipath Image Segmentation

Dataset :
We have considered ATLAS dataset which has 300 anat and their respective lesion file. We have decided to split the data into 20 train images, 20 test images and 260 dataset
The shape of images are : 193*229*193
As we have the files in nifty format, we have converted the dataset into numpy format.
An alternative to this approach is to use nibabel library to load and read images, but we have used the numpy format for our project.
So our preprocessing code has saved numpy format of train_images, their labels, and set of similar_images after finding the minimum Euclidean distance between two images.



Methodology Used :
In our methodology we have taken one train file and found the similar image from the whole dataset and these two images goes as two inputs to our multipath approach.
Both the images are down sampled to an image size of 48*57*48 and 128 channels.
We have incorporated depth of 4 in our Unet, where the channels increases from 1 to 32 to 64 to 128.
Once both the encoders are run for two different images, downsampling features are averaged and fed as input to decoder.
The up sampling or the decoder blocks uses the converse transpose to increase the size of the image while decreasing number of channels.

￼


Results
For Training the dataset we have trained the model based on 20 train images.
For every epoch, the whole methodology is run for one image from train file.
We will be comparing the cross validation for a single path 3D unet model to a multipath 3D unet Model.
We are comparing the results based on Dice accuracy which in our case is 1-Dice Loss.
For both the epochs, we have run our code for 20 individual images through the whole architecture and then finding the average dice loss over all the epochs.




Single Path 3D Unet
MEAN DICE ACCURACY = 1- MEAN DICE LOSS = 1 - 0.0838 = 0.916 = 91.6 % DICE ACCURACY

￼




Multi-Path 3D Unet

MEAN DICE ACCURACY = 1- MEAN DICE LOSS = 1 - 0.047 = 0.953 = 95.3 % DICE ACCURACY

￼


￼
