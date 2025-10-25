This code was based on the repository: https://github.com/shreydan/VisionGPT2/tree/main

This code is for Flickr30k dataset. New datastets can be used by changing the code in the main.py and eval.py files

create enviroment using the requirements.txt file

download the dataset and change the directory names in the main.py file and eval.py file accordingly

run the main.py script to train the model: python main.py

run the eval.py for producing the captions for the test set: python eval.py

calculate the metrics using metrics.py by comparing generated labels and ground labels: python metrics.py
