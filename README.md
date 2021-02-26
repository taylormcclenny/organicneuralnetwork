# organicneuralnetwork
Background: For the most part, advancements and foci in Artifial Intelligence are getting increasingly contrived. These "increasingly contrived" models are great for their narrow tasks, but I would like to see more adaptive models, inspired by basic principles of biological adaptation and the versatile architectures and guiding principles of the brain and nervous system.

This Visual ONN ("Visual Organic Neural Network") is a branch of my work in a greater body of work in achieving my goal to see more adaptive, more "humnan-like" Neural Networks. My series of ONN architecures are a return to first principles, building new NN architectures from foundational Neuroscience principles - and no graident descent.

In beginning this project I started with a blank script and clear mind, not borrowing concepts from any existing models. Any time I was stuck with how to solve a problem, I researched how the brain/nervous system solved this problem, and used that as my guide to solution. Over time, as Aritificial Intelligence/Neural Networks borrow from basic principles of the brain, overlap between my principles and existing principles naturally emerged. This has been a great learning lesson in the first principles of Neuroscience and Artificial Intelligence.

Below I will discuss some basic differences and similarities between my ONN principles and those commonly seen in more mainstream Artificial Intelligence principles.

Perhaps the most basic principle and difference is that which a neuron uses to determine if it "fires" (sends signal). Many existing NN propogate positive and negative values achieved by multipling inputs by weights and biases, then "squishing" the output to between 0 and 1 or -1 and 1. In biology, and in most neurons of my ONN, neurons are either "Excitatory" or "Inhibitory" - their signal output is either positive (excitatory) or negative (inhibitory). Receiving signal from many pre-synaptic (or "upstream") neurons, each neuron sums these positive and negative signals (in the axon hillock) to determine if they fire. If their threshold for firing - that is to say, the inhibitory signals outweighed the excitatory signals - they do not fire, propogating no signal. This principle is illustrated below, showing signal propogation through the retina of the eye - from a grouping of Rods to a coupling of Bipolar Cells to a single Retinal Ganglion Cell.

![basic_signal_propogation](https://github.com/taylormcclenny/organicneuralnetwork/blob/master/basic_signal_propogation.png)

Below is a print out of what the AI is seeing as it looks over an image. With each signal propogation, it decides how to move its gaze to see more of what's interesting in the picture.

![field_of_vision in terminal](https://github.com/taylormcclenny/organicneuralnetwork/blob/master/onn_diagram.png)

GOAL:  This AI looks around simple images, focusing on what's interesting. It has a small "field of vision", mimicing our (human) focal and peripheral vision. This serves as inputs to simply let the AI know (1) if it's looking at something and (2) where it should look next to focus on what's interesting. This is the first step in our own (human) visual process. The eye sends signals to the occipital lobe where it determines if it's "looking at something of interest". Based on the interest and basic signal characterisitics from the various neurons of the retina, the occipital lobe decides where to next send (like a dispatcher) the signal (for identification, movement tracking, etc.)

DISCLAIMER:  This is an ongoing and rapidly developing project. The majority of my time is spent researching Neuroscience (the various neuron layers of the eye, the ocular nerve, the occipital lobe, their feedback, and the first principles that drive them) and experimenting. As such, I am currently spending little time optimizing this code. Everything is currently focused on R&D and gaining observable, first principle understanding.

# Folders & Files
research/ - Contains most pertinent, distilled information from research papers, diagrams, images, and videos.

images/ - Contains various images used for testing/training.

/generate_onn.py - Generates the "brain map" of the Neural Network (as a Python dictionary or JSON file). It details how the various layers of the eye (rods & cones, bipolar neurons, ganglion neurons, etc.) begin their initial interaction with each other. This brain map is fed into the "signal generator" aka. "run_onn.py" where a signal is sent through the brain map.

/run_onn.py - Creates a "field of vision" to view a small portion of the image. Runs the signal through the brain map. Returns a direction to update where the "field of view" should look next.

  It starts by loading an image into memory as it will reference the image many times in training (this image can be any size). 

  The main() loop runs through: (1) establishing a "field of vision" (the AI's focal and peripheral vision) on the image, (2) running the next "layer" of the brain map, (3) assigning each neuron to a parallel process, (4) computing that neuron's signal, (5) looking up it's post-synaptic neighbors, (6) sending the signal to the cooresponding post-synaptic neuron ("neighbor"), (7) repeating until reaching the "Direction Deciding" neurons, and (8) sending back a signal for which direction the "field of vision" should move for the next observation and signal pass.

  It is very important to understand that it "observes" only a small portion the image at any given time. The goal of this project is to mimic the human eye/brain and it's principles of observation and interest. It's important to understand that we (humans) actually see and compute VERY little of what is observable to us. Reason being, our brains couldn't handle computing 100% of the information we could supply it -- And, what might kill you or feed you (or reproduce with you..) often occupies a very small percentage of your total observable environment. For this reason our brains have spent a great deal of evolutionary energy learning to focus and build interest in things that are then pieced together for a coherent "image" in our brain.
