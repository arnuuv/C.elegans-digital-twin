C.elegans nematode

1. What OpenWorm has done yet

a. Built a neural model (c302) -

- Digital models of different subsets of the connectome (302 neurons).
- Simulated neuron activity with ion channels (via NEURON)

b. Body Model (Sibernetic) -

- Physics engine simulating soft-body physics in fluid environments
- Models of muscle contraction affecting movement

c. Connectome Mapping -

- Integrated full C.elegans connectome
- Neuron classification and naming

d. Visualization -

- 3D anatomical rendering with Geppetto
- Worm Browser for navigating neurons, muscles, etc

e. Development Modelling (DevoWorm) -

- Partial lineage tracking during development

2. What OpenWorm has left

a. Full Body + Brain Integration

- No fully integrated brain → muscle → motion loop that behaves like a real worm

b. Behavioral Simulation

- No autonomous behavior such as foraging, chemotaxis, escape reflexes.

c. Validation Against Biology

- Limited experimental comparison (e.g. how closely do simulated neurons mimic real ones?)

d. Real-Time Interactive Sim

- No real-time control or fully interactive sim yet

e. Developmental Simulation

- DevoWorm is still exploratory; not a full embryo-to-adult growth sim

f.Full Lifecycle Modeling

- No digital twin for developmental stages (L1 to adult)

g. Higher-Level Functions

- No memory, learning, or adaptive behavior implemented

To do -

Tools we need to looks at

1. Worm Atlas - https://www.wormatlas.org/
2. Neuron classes and windows - https://www.wormatlas.org/hermaphrodite/nervous/mainframe.htm
3. NEURON simulator - https://www.neuron.yale.edu/neuron/
4. c302 - simulate a small neural network - https://github.com/openworm/c302
5. Sibernetic - https://github.com/openworm/sibernetic (simulate body and movement)
6. Geppetto - https://github.com/openworm/org.geppetto (visualize neuron/muscle activity in 3D.)

Building our own model

1. Two Options - either simulate just the brain or create a brain body loop

- Connect neural outputs from c302 → muscles → body motion (via Sibernetic).
- Add sensory inputs (nose touch, light, heat).

2. Expand to Behavior

- Implement simple reflexes (e.g. move forward, turn, escape)
- Tune based on real worm behavior videos

Neural Network Structure

1. Each neuron = one ANN node.
2. Use the connectome graph as ANN architecture.
3. Inputs = sensory neurons
4. Outputs = motor neurons
5. Hidden = interneurons

6. We should train on tasks like:

- Move forward
- Turn left or right on light input
- Avoid obstacles

2. We could use feedforward or RNN or GNN's.

Example -
Input: Nose touch (1 or 0)
Connectome-shaped neural network
Output: Muscle activation -> Move away

Data -

1. ion channels - http://docs.google.com/spreadsheets/d/1SSgYO2cn29-C91boU4wqvK5jBOSWi7a1Hz1bzTeikzU/edit?gid=0#gid=0 (useless rn)
2. neurons - https://www.wormatlas.org/neurons/Individual%20Neurons/Neuronframeset.html

we should map neurons to stimulus. and they should output motor (last few neurons starting with V for movement in neuron sheet)Data Type Format
Neuron list (names & types) - CSV or JSON
Connectivity (from/to/weight/type) - Edge list (CSV or pandas DataFrame)
Stimulus mapping (input neurons) - Dict or CSV
Motor neuron list (output neurons) - List or CSV

all data of neurons+mapping -> https://www.wormatlas.org/neuronalwiring.html

stimulus mapping -> it has 302 neurons but only 60 sensory neurons.

Input neurons 5-10 sensory neurons AVM, ALM, ASH, AWA, AFD
Hidden neurons All interneurons AVA, AVD, AIA, RIM, etc.
Output neurons 5-10 motor neurons VA, VB, DA, DB, DD, VD
Edges (connectivity) From connectome file AVM → AVA (weight = 3, gap)
