# CHATBOT DARE TO DONOR WITH ANN MODEL 
Hi! Welcome to Dare To Donor Chatbot. This bot is designed to provide information about blood donation, donor procedures, blood requests, and other related topics.

## How to use
1. Clone this repository.
2. Install the dependencies by running the command `pip install -r requirements.txt`.
3. Run the application with the command `python chatbot.py`.

## Ways of working

### Data Preprocessing:
First, the intents.json file contains various patterns of questions and responses provided by users and bots.
Each question pattern is converted into a list of words and then converted into basic verb forms using WordNet Lemmatizer.
The result is a collection of documents, where each document consists of a list of words and their corresponding tags.

### Model Creation:
The list of words and tags is passed to the machine learning model.
The model uses an artificial neural network with several Dense layers.
This model is trained to classify question patterns into appropriate tags (intent) using one-hot encoding labeling.
Once trained, the model is saved as the chatbot_model.h5 file.

### Model Usage:
The model can be used to predict the intent of a question given by the user.
The use of this model is implemented in the predict_class function.
The prediction results are then used to select the appropriate response from the intents.json file with the get_response function.

### Interaction with Users:
There is an infinite loop that requests input from the user.
Users can enter questions or commands. If the user enters "exit," the program exits the loop and terminates the application.
Each user input is processed by the model to predict intent, and an appropriate response is displayed.
