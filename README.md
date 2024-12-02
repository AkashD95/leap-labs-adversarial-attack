# leap-labs-adversarial-attack
 model agnostic adversarial attack

 - Using MNIST pre-trained models as the test model
 - Run a white box adversarial attack (control experiment) using the torchvision tutorial. Use the evaluation code for the next steps. -> gfsm_example.ipynb
 - Generate a basic UI that will run the evaluation on an image -> ui folder, run using app.py

 - Then move onto class specific attack that humans cannot recognise but MNIST will misclassify. -> model training in train_ae.py
    -Train an auto encoder  on the train section of the pytorch MNIST data set (leftover for downstream evaluation) to create a latent space. Train a decoder to go back to the image afterwords. Evaluate performance. Train until the workflow generates realistic images. -> Done (could be better)
    -Select the class you want to the adversarial attack to mimic (e.g for MNIST lets say 5)
    -Run the selected class through the encoder and extract the latent variables of the latent variables. -> to do 
    -Generate a latin hypercube design using the space in the space between example images you want MNIST to misclassify and the selected class. -> to do
    -Run the latin hypercube designs through the decoder and evaluate on MNIST to see which class they are assigned. At this time give these samples to a humans to also classify -> to do 
    -Do rounds of active learning using a simple objective functions e.g score = does MNIST misclassify to the selected class + does the human classify to the correct class -> to do
    -Using this to get information on how to peturb the latent space in a generalisable way. -> to do
    
    
    