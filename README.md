# organicneuralnetwork
Advancements in NN are getting increasingly contrived. My ONN ("Organic Neural Network") is a return to first principles, building a new NN architecture from foundational Neuroscience principles - AND NO GRADIENT DESCENT.

GOAL:  This AI looks around simple images, focusing on what's interesting. It has a small "field of vision", mimicing our (human) focal and peripheral vision, this serve as inputs to simply let the AI know (1) if it's looking at something and (2) where it should look next, to focus on what's interesting. This is the first step in our own (human) visual process. The eye sends signal to the occipital lobe where it determines if it's "looking at something of interest". Based on the interest and basic signal characterisitics from the various neurons of the retina, the occipital lobe decides where to next send (like a dispatcher) the signal (for identification, movement tracking, etc.)

DISCLAIMER:  This is an ongoing and rapidly developing project. The majority of my time is spent researching Neuroscience (the various neuron layers of the eye, the ocular nerve, the occipital lobe, their feedback, and the first principles that drive them) and experimenting. As such, I am currently spending little time optimizing this code. Everything is currently focused on R&D and gaining observable, first principle understanding.

# Folders & Files
research/ - Contains most pertinent, distilled information from research papers, diagrams, images, and videos.

images/ - Contains various images used for testing/training.

/generate_onn_v2.py - Generates the "brain map" of the Neural Network (as a Python dictionary). It details how the various layers of the eye (rods & cones, bipolar neurons, ganglion neurons, etc.) being their initial interaction with each other. This brain map is fed into the "signal generator" aka. "run_onn.py" where a signal is sent through the brain map.

/run_onn.py - Creates a "field of vision" to view a small portion of the image. Runs the signal through the brain map. Returns a direction to update where the "field of view" should look next.

  It starts by loading an image into memory as it will reference the image many times in training (this image can be any size). 

  The main() loop runs through: (1) establishing a "field of vision" (the AI's focal and peripheral vision) on the image, (2) running the next "layer" of the brain map, (3) assigning each neuron to a parallel process, (4) computing that neuron's signal, (5) looking up it's post-synaptic neighbors, (6) sending the signal to the cooresponding post-synaptic neuron ("neighbor"), (7) repeating until reaching the "Direction Deciding" neurons, and (8) sending back a signal for which direction the "field of vision" should move for the next observation and signal pass.

  It is very important to understand that it "observes" only a small portion the image at any given time. The goal of this project is to mimic the human eye/brain and it's principles of observation and interest. It's important to understand that we (humans) actually see and compute VERY little of what is observable to us. Reason being, our brains couldn't handle computing 100% of the information we could supply it -- And, what might kill you or feed you (or reproduce with you..) often occupies a very small percentage of your total observable environment. For this reason our brains have spent a great deal of evolutionary energy learning to focus and build interest in things that are then pieced together for a coherent "image" in our brain.
