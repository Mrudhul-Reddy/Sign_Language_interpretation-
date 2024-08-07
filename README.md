# Sign_Language_interpretation

This folder has various python codes and libraries which are useful for collecting data(images) of sign language and storing it in the desired location. Trraining of the model can either by done by using code or an open source platform called Teachable Machines. The main function of the model is to train the data using tensorflow and openCV and test it. 
The requirements text file by the name req.txt is given.
Install python and in the terminal window of the editor install all the libraries in the req.txt file.
Run the DataCollection.py to collect the data of the sign language you want make model of, like ISL(Indian Sign Language), ASL(American Sign Language).
Run the Train.py or use Teachable Machines web application to train the model on the images collected.
Run Test.py to see the output od the model. The output can seen by running the model as it is, as the model has already been given in the Trained directory.
Make sure to change the path of respective directories, where to store images, where is the model stored, etc.
This project is done by classifying images, this project can further be improved by integrating videos for action signs in the sign language. This can also be used in hardware like raspberry pi for hardware implemntation.
