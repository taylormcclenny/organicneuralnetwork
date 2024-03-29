# organicneuralnetwork
Background: For the most part, advancements and foci in Artifial Intelligence are getting increasingly contrived. These "increasingly contrived" models are great for their narrow tasks, but I would like to see more adaptive models, inspired by basic principles of biological adaptation and the versatile architectures and guiding principles of the brain and nervous system.

![field_of_vision in terminal](https://github.com/taylormcclenny/organicneuralnetwork/blob/master/onn_diagram.png)

This Visual ONN ("Visual Organic Neural Network") is a branch of my work in a greater body of work in achieving my goal to see more adaptive, more "humnan-like" Neural Networks. My series of ONN architecures are a return to first principles, building new NN architectures from foundational Neuroscience principles - and no complex mathematics or gradient descent.

In beginning this project I started with a blank script and clear mind, not borrowing concepts from any existing models. Any time I was stuck with how to solve a problem, I researched how the brain/nervous system solved this problem, and used that as my guide to solution. Over time, as Aritificial Intelligence/Neural Networks borrow from basic principles of the brain, overlap between my principles and existing principles naturally emerged. This has been a great learning lesson in the first principles of Biology and Artificial Intelligence.

Below I will discuss some basic differences and similarities between my ONN principles and those commonly seen in more mainstream Artificial Intelligence principles.

Perhaps the most basic principle and difference is that which a neuron uses to send signal. Many existing NN propogate positive and negative values achieved by multipling inputs by weights and offsetting with biases, then "squishing" the output to between 0 and 1 or -1 and 1. Firstly, In biology, and in most neurons of my ONN, neurons either "fire" (send signal) or don't (rather than propogating negative values). Secondly, they are either "Excitatory" or "Inhibitory", meaning, upon firing, their neurotransmitters either excite or inhibit their post-synaptic (or "downstream") neuron neighbors. If the summed signals are inhibitory the post-synaptic (receiving) neuron simply doesn't fire at all. It sends no signal, and over time, degenerates due to a lack of stimulus. If the summed signal are excitatory, the neuron carries out its function, passing along it's either excitatory or inhibitory signal.

This principle is illustrated below, showing signal propogation through the retina of the eye - from a grouping of Rods to a coupling of Bipolar Cells to a single Retinal Ganglion Cell.

![basic_signal_propogation](https://github.com/taylormcclenny/organicneuralnetwork/blob/master/basic_signal_propogation.png)

Here 9 Rods concurrently send their signals (all, always positive values) to 2 Bipolar Cells, which in turn, concurrently send their signals to 1 Ganglion Cell. Depending on the strength (or sum) of the the excitatory and inhibitory signals from the Bipolar Cells, the Ganglion will fire or do nothing. In this example, the excitatory signal outweighs the inhibitory signal and the Ganglion fires, sending it's signal through the occular nerve into the occipital lobe, stimulating multiple entry points.

Different cells have different functions - as we saw with the coupling of the Bipolar Cells (as Rods always send postive signals) - but this simple summation of excitatory and inhibitory signals and this principle of "fire or don't" is foundational to how all cells work in both Biology and the ONN.

Below is a print out of what the AI is seeing as it looks over an image. With each signal propogation, it decides how to move its gaze to see more of what's interesting in the picture.

![scanning_image](https://github.com/taylormcclenny/organicneuralnetwork/blob/master/run_onn_v1.gif)


Now understanding a couple simple principles of the ONN, let's look at the first major goal/task of this Visual ONN and a sample architecture to achieve this goal.

GOAL:  This AI looks around simple images, focusing on what's interesting. It has a small "field of vision", mimicing our (human) focal and peripheral vision. This serves as inputs to simply let the AI know (1) if it's looking at something and (2) where it should look next to focus on what's interesting. This is the first step in our own visual process. The eye sends signals to the occipital lobe where it determines if it's "looking at something of interest". Based on the interest and basic signal characterisitics from the various neurons of the retina, the Occipital Lobe decides where to next send (like a dispatcher) the signal (for identification, movement tracking, etc.).

Another major difference (and similarity) between this sample Visual ONN and more traditional Convolutional Neural Networks ("CNN"), is that the Visual ONN is only looking at a small portion of the image at any given time. This is very similar to the concept of filters in CNNs, however, there is no contrived, or human built-in edge detection or image manipulation, nor is there a human provided systematic sweep over the full image. But rather, the AI takes in raw (but Normalized) values, which go straight to the Rods and Cones of the Visual ONN, and the AI, upon signal reaching the Occipital Lobe, decides where to look next.

CURRENT STATUS: I'm working on 2 major pieces: 

(1) Learning/Growth - The example architecture shown above is the "starting brain" for the Visual ONN. Some portions have extra connections (as is seen in very, very early childhood) and some portions have much fewer neurons than is expected. This is a balance of having extra neural connection resources - as very early on, it's faster to have and degenerate, than to not have and generate - and not having too much human-contrived network that deeper occipital neurons have to navigate.

(2) Horizontal Cells/Naturally Increasing Contrast - In your eyes, Horizontal Cells lie between Rods/Cones and Bipolar Cells. Horizontal Cells run.. horizontally between many Rods/Cones. They are all, always inhibitory. The reason for this is that they dampen low to moderate signal intensities. So, if a region of Rods/Cones is receiving a gradient of low-to-moderate-to-high signal intensities, the lower intensities are spread accross neighboring Rods/Cones, so that only the very high signal intensities are passed on to the Bipolar Cells.

-----------
DISCLAIMER:  This is an ongoing and rapidly developing project. The majority of my time is spent researching Neuroscience (the various neuron layers of the eye, the ocular nerve, the occipital lobe, their feedback, and the first principles that drive them) and experimenting. As such, I am currently spending little time optimizing this code. Everything is currently focused on R&D and gaining observable, first principle understanding.
-----------

# Folders & Files
research/ - Contains most pertinent, distilled information from research papers, diagrams, images, and videos.

images/ - Contains various images used for testing/training.

/generate_onn.py - Generates the "brain map" of the Neural Network (as a Python dictionary or JSON file). It details how the various layers of the eye (rods & cones, bipolar neurons, ganglion neurons, etc.) begin their initial interaction with each other. This brain map is fed into the "signal generator" aka. "run_generation.py" where a signal is sent through the brain map.

/run_generation.py - Creates a "field of vision" to view a small portion of the image. Runs the signal through the brain map. Returns a direction to update where the "field of view" should look next.

It starts by loading an image into memory as it will reference the image many times in training (this image can be any size). 

The main() loop runs through: (1) establishing a "field of vision" (the AI's focal and peripheral vision) on the image, (2) running the next "layer" of the brain map, (3) assigning each neuron to a parallel process, (4) computing if that neuron should fire or not, (5) looking up it's post-synaptic neighbors, (6) sending the signal to the cooresponding post-synaptic neuron ("neighbor"), (7) repeating until reaching the "Direction Deciding" neurons, and (8) sending back a signal for which direction the "field of vision" should move for the next observation and signal pass.

It is very important to understand that it "observes" only a small portion the image at any given time. The goal of this project is to mimic the human eye/brain and it's principles of observation and interest. It's important to understand that we (humans) actually see and compute VERY little of what is observable to us. Reason being, our brains couldn't handle computing 100% of the information we could supply it -- And, what might kill you or feed you (or reproduce with you..) often occupies a very small percentage of your total observable environment. For this reason our brains have spent a great deal of evolutionary energy learning to focus and build interest in things that are then pieced together for a coherent "image" in our brain.
