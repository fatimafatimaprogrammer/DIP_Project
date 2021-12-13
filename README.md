Face Mask and Drowsiness Detection

Abstract—After COVID many changes in life have been observed and many new things are now mandatory for us to ensure our safety and safety of our loved ones like wearing a face mask and maintaining a social distance of 6ft. Implementation of COVID SOP's and making sure that everyone follows them is not an easy task and it is almost impossible to ensure it manually. We can use technology to help us implement these SOP especially inside classrooms and institutions. Almost all institutions have cameras like we have cameras inside offices, classrooms, etc. We can use these cameras not only to ensure that everyone follows SOPs but also to increase efficiency in workplace, schools, colleges and universities. We propose to develop an application that will provide features like mask detection, distance detection, drowsiness detection and distraction detection. Our program would take images captured by cameras and use them to detect either an individual is wearing a facemask or not. Then we would use this application to detect if that individual is maintaining social distance from fellows and then comes two modules related to mood/emotions. We will detect if an individual is feeling drowsy or not and lastly if that individual is active, he is focused on work or is distracted and looking somewhere else other than an area of focus.

I. Introduction
Digital Image Processing, or DIP for short, is all about manipulating an image to improve its quality for various purposes. It involves the use of a digital computer and complex algorithms to offer sophistication and an accurately processed image. In DIP, we provide an image input to the program and the output can be an image, objects, edges etc. depending on what we need.
For our project, we used a few tools to process a video frame by frame and then detect things like:
•	Face Masks
•	Drowsiness
The first step in all of these modules is Face Detection. With the help of TensorFlow, which is a library for machine learning, and Keras, an API designed for artificial neural networks, we have trained a model to first detect faces and then proceed onto the specific detection. We will discuss their applications in detail.

I.I. TensorFlow:
It is a machine learning library released by Google for fast computing. The applications of this library spread over development, research, and production fields, and it includes many deep learning models which we can experiment with. The numerical computations can be described as graph operations and data flow in them. In a graph, the data between nodes is called tensors; these nodes are the operations, while the edges are the actual data. It is used in various applications:
•	Recognition – voice, image, objects etc.
•	Detection – motion, distance, face etc.
•	Text-based Applications – processing of text messages, tweets, face book comments etc.
In our project, we have made use of Detection, mainly detection in a video. Our detection environment is built in the following steps:
1.	Environment setup
2.	Acquiring a dataset of videos for testing and training
3.	Training of the model
4.	Testing
After these steps, our model is ready to detect faces, face masks and drowsiness.

I.II. Keras:
Keras is a user-friendly deep learning library which evaluates models to make predictions and correct accuracy. It supports recurrent and convolutional neural networks (CNN) which are really helpful in analyzing visual imagery. It involves the following steps:
1.	Acquire data from a dataset and load it
2.	Define and compile keras model as a sequence of layers
3.	Execute the model on some data
4.	Evaluate the performance of the model
5.	Adapt the model to make predictions for new data

I.III. Deep Neural Networks (DNN):
DNN is a machine learning technique which trains models in the way a brain learns. Because a computer doesn’t know what information it is being provided with when we use sound or images, the computer needs to use its DNN for data recognition and processing. It then makes decisions about the type of data provided to it and gains feedback to improve its decision-making skills.
DNN can be used for:
•	Image segmentation and classification
•	Object, Person and Face detection
•	Depth and Pose Estimation
We used OpenCV DNN module in our project to process real-time video for object detection and classification. We chose DNN because it supports TensorFlow framework and also because its inference time is fast, so for processing real-time videos we can get good inference. [1]
There also are many basic python libraries used. So we will not go in too much detail. Onto our project Modules:


II. Module 1: Face Mask Detection
II. I. Requirements
•	TensorFlow >= 1.15.2 [2]
•	Keras == 2.3.1 
•	Imutils == 0.5.3
•	Numpy == 1.18.2
•	opencv-python == 4.2.0.*
•	matplotlib == 3.2.1
•	scipy == 1.4.1

II.II. Dataset
Our dataset is divided into two categories which includes 1915 images with mask and 1918 without mask. [3]
II.III. Face and Face Mask Detection:
II.III.I. Data Preprocessing:
Firstly, we defined two lists named Data and Labels. Data list contains the images and Labels contains the labels either image in data list is with mask or without mask, and then we loaded the data in those lists by specifying the path.
Now when we have data in Data list, it is in numerical form. But the categories in Labels list is in alphabets, so that was not useful for us. We needed to first binarize the categories using LabelBinarize coming from sklearn.preprocessing, and then converting those categories with mask and without mask into variables.
 
Lastly, we converted the Data and Labels lists into array using numpy arrays,  because deep learning models work with arrays.
II.III.II. Splitting the data:
We split the data into test and train test using train_test_split method with the test_size = 0.20.
  Here, the data preprocessing is completed.

II.III.III. Base Model:
We used MobileNetv2 in this particular model for detecting face masks, and in order to use the MobieNetv2 we have to perform the preprocess_input function,
 
on the input images which can be imported from tensorflow.keras.preprocessing.image.
The MobileNetV2 architecture is computationally efficient, thus it makes it easier to deploy the model to embedded systems. Although it is a little bit less accurate than convolutional neural network, but for our use case it will surely serve the purpose.
In the training model python file, we initially had  Epochs = 20, Bs=backsize=42  [3] and learning rate = 1e-4 because we had to make sure that the initial learning rate should be low. When learning rate is low, loss is calculated properly which means that we’ll get the accuracy very soon.

II.IV. Training:
 
We are training the mask detection model like explained in illustration above with a small change in it instead of CNN we used MobileNetV2 to extract features at second stage. First we feed the input image in form of arrays to input as deep neural network algorithm works with arrays. Then we used MobileNetv2 as a base Model over the Head model instead of Convolutional network because they are efficient and performs best in our use case. After that we are maxpooling with the size of 7x7 and flatten it and fully connected it and generated output. [5]

II.IV.I. Run and View the Model:
  

II.IV.II. Plotting accuracy of model:
  

II.IV.III. Use model in real time camera: (detect_mask_video.py):
To test our model, we use real time camera to capture videos and apply our algorithm on it. Facenet in detect mask.py source file is reading the face detection models as we gave path of those models
 
and passing them to readNet method of DNN (deep neural network) model which comes from cv2.Cv2 developed recently this DNN with a lot of mini useful methods in them.
 
Face net is detecting the face and mask detection is done through our trained model using MobileNetv2 as a base model. Once we have loaded both the models we now need to load the camera for that and using cv2.videostream function we live stream and get the video and apply our trained model to get the results.

II.V. Result: 
 
 


III. Module 2: Drowsiness Detection

III.I. Requirements
•	TensorFlow
•	Keras
•	OpenCV/cv2
•	Numpy

III.II. Model:

III.II.I. Dataset
Dataset [6] is divided into two parts one is for training and other is for validation. Each part consists of two categories named “open” and “close”. These consist of images of eyes in open and closed form. The training data set consist of 1234 images while the validation set has218 images.

III.II.II. Base Model:
The architecture of Model is Convolution Neural Network (CNN). CNN was used for this model as CNN performs extremely well for Image processing. CNN consist of an input layer and an output layer and a hidden layer. This hidden layer can be of one layer or it can have numerous layers in it. In this model file we have 3 convolution layers and after each convolution layer, there is a max pooling layer which is used to converge the dimensions of the output. After that the model is being flattened and a model is generated using model.fit_generate() method of keras. After training, the trained model is being saved and is being loaded in main file to be used to detect drowsiness in real time.
 


III.III. Main File

III.III.I. Video Capturing
Video is being captured in real time by using python “cv2” library. Here we are making an object of VideoCapture() that then uses .read() method to capture video frames. 

III.III.II. Obtaining Images from Video
The method .read returns a tuple that include a Boolean value indicating that either frame was read and captured correctly and a frame as the second item of tuple.
  	III.III.III. Extraction of ROI
From the frames obtained from the above mentioned method, we are then extracting ROI. For this purpose we are using cv2’s “CascadeClassifier()” method. We are using three different haarcascade files. One for facial detection and two for left and right eye detection. For this purpose we are creating three different object of cascade classifier one for each file and gives them a grayscale image. We obtain gray scale image from by using cv2.cvtcolor() method. Each cascade classifier provides us with ROI by providing us 4 return values that include (x,y) coordinates of starting point of that ROI and (w,h) of that ROI.
 

	III.III.IV. Preprocessing of ROI
After we have obtained ROI using cascade classifiers now, we will do some preprocessing on ROI so that it can be given to the model for prediction. Since we are making a drowsiness detection system our main focus is on eyes that either eyes are close or open. We first store eye images in an frame using (x,y) and (w,h) values that were output of our cascade classifiers. We than convert them in grayscale and resize the ROI using cv2.resize() method to convert it in dimensions of the training set images so that it can easily be classified by our model. After this, the images are normalized by dividing them with 255. 
 




	III.III.V. Getting Prediction
Trained model for prediction was loaded at start using “Keras” loadmodel() method and it’s object was stored in a variable. After all the preprocessing, the eye images are given to the model object. This model object returns prediction values based on image. Based on the value of prediction, we then give images a label of open or close.


	III.III.VI. Output
After getting prediction and adding label to the images we use them to output result to the frame in real time by using cv.putText() method. If both or any one eye is open, then output is open i.e., person is awake. If both eyes are closed, output is closed i.e., that the person is sleeping. When both eyes are closed, timer counts time and add score to indicate for how long a person have been sleeping.

III.IV. Result
 
 
 
  

IV. Future Trends and Applications
There are various future trends of face mask and drowsiness detection. 
As stated in the abstract, the COVID-19 virus has made it really important that each and everyone follows the SOPs. We can feed this algorithm in CCTV cameras of public places, schools, universities, banks, offices etc. to ensure following of SOPs. Integrating facial recognition, the cameras can automatically detect defaulters so that the authorities don’t have to do it manually. With distance detection, we can ensure social distancing measures as well. It can also be used in smart-city innovation. 
Mask detection can also be used in hospitals, where masks are necessary in operating rooms and wards. The detector would trigger alarm if it detects a hospital staff without a mask in these areas.
Similarly, the drowsiness detection module can be used in classrooms for detecting sleepy and inattentive students. It can also be useful in detecting sleepy drivers on the road to prevent accidents. 
Furthermore, we can implement CNN based models for improved accuracy and correct predictions.

References

[1] 	"What is a Deep Neural Network," https://www.oticon.com/blog/what-is-a-deep-neural-network-dnn.
[2] 	"https://www.tensorflow.org/install/pip#windows_1".
[3] 	"https://github.com/balajisrinivas/Face-Mask-Detection".
[4] 	"https://www.baeldung.com/cs/epoch-neural-networks".
[5] 	"https://paperswithcode.com/method/max-pooling".
[6] 	"https://www.kaggle.com/serenaraju/yawn-eye-dataset-new".





