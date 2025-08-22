for the data in connectivity-data-readme


Column 1 - neuron1 - Presynaptic neuron (signal sender)
Column 2 - neuron2 - Postsynaptic neuron (signal receiver)
Column 3 - Type - Type of synapse (kind of connection between neurons)
Column 4 - Nbr - Number of synapses between the given neuron pair.

| Code   | Full Name                   | Description                                                                 |
| ------ | --------------------------- | --------------------------------------------------------------------------- |
| **EJ** | Electrical Junction         | Also called **gap junctions** – direct, fast connections                    |
| **Sp** | Synaptic (chemical)         | **Standard excitatory chemical synapse**                                    |
| **R**  | Reciprocal                  | Possibly **mutual connection**, but sometimes refers to other nuanced forms |
| **Rp** | Possibly Reciprocal synapse | Often used for **bidirectional** chemical connections                       |
| **S**  | Synapse                     | A general **chemical synapse** (often used inconsistently)                  |
| **Cs** | Unknown/complex synapse     | Not always clearly defined                                                  |
| **NMJ**| NeuroMuscular Junction     | synapse where a motor neuron connects to the muscle cell                    |

Dont use NMJ cuz were only doing brain rn.



It must be noted that at polyadic synaptic sites, not all “send-poly” were faithfully labeled as such in White et al, 1986. Some pre-synaptic connections were labeled simply as “sends”. Reconciliation of chemical synapses did not previously distinguish between send from send-poly and receive from receive-poly. In this new reconciliation, the total number of send and send-poly is equal to the total number of receive and receive-poly (S+Sp=R+Rp). Every documented synapse is now listed in this Table, both with respect to the sending neuron and with respect to the receiving neuron(s).

Polyadic synapses are those where more than one postsynaptic partner receive input at one release site. Postsynaptic partners can include other neurons, muscle arms (muArm) and rarely hypodermis (synapses onto hypodermis are not included in this dataset analysis). A. A synapse with one postsynaptic partner. B,C. Polyadic synapse types. D. A dyadic synaptic site. In C. elegans, synapses are identified by dark pre-synaptic tufts at the region of vesicle release. No specialization is generally visible for post-synaptic elements. In this electron micrograph, the pre-synaptic neuron VA4 appears directed towards both VD4 and a muscle arm. This type of synapse is labeled as send-poly (Sp) and receive-poly (Rp) in the connectivity data.


Provided is a compilation of an updated version of C. elegans wiring diagram (280 nonpharyngeal neurons (CANL/R were excluded since they have no obvious synapses), covering 6393 chemical synapses, 890 electrical junctions, and 1410 neuromuscular junctions). Pivotal works published by White et al, 1986 and Hall and Russell, 1991 had provided neuronal circuitry in the head and tail, but lacked connection details for 58 motor neurons in the ventral cord of the worm. Most of the missing data for this region is now compiled by using original electron micrographs (EM) and handwritten notes from White and co-workers. The dorsal side of the worm around the mid-body was not previously documented. Using original thin sections prepared by White et al, 1986, new EM images were generated and neuron processes of animal “N2U” in this region were reconstructed. The new version of the wiring diagram incorporates original data from White et al, 1986, new reconstructions, as well as updates based upon later work (Hobert and Hall, 1999; and Durbin, R. M., 1986, Achacoso and Yamamoto W.S., 1991, Chen. Inconsistencies within the data were reconciled by checking against original EM and handwritten notes from White and co-workers. Over 3000 connections, including chemical synapses, electrical junctions, and neuromuscular junctions were added and/or updated from the previous version. Due to rather sparse sampling of data along lengths of the sub-lateral, canal-associated lateral, and mid-body dorsal cords, connectivity ambiguities for a select few neurons remain.

The current wiring diagram is considered self-consistent under the following criteria: (I) A record of Neuron A sending a chemical synapse to Neuron B must be paired with a record of Neuron B receiving a chemical synapse from Neuron A. (II) A record of electrical junction between Neuron C and Neuron D must be paired with a separate record of electrical junction between Neuron D and Neuron C. From our compilation of wiring data, including new reconstructions of ventral cord motor neurons, we applied the above criteria to isolate records with mismatched reciprocal records. The discrepancies were reconciled by checking against electron micrographs and White and coworkers’ lab notebooks. Connections in the posterior region of the animal were also cross-referenced with reconstructions published by Hall and Russell, 1991. Reconciliation involved 561 synapses for 108 neurons (49% chemical “sends”, 31% chemical “receives”, and 20% electrical junctions).

The included file is the updated wiring diagram (as of 2/2/2006) listing the number of connections by neuron, its synaptic partner, and by type of synapses (incoming chemical synapse or “receives”, outgoing chemical synapse or “sends”, electrical junction and neuromuscular junction (NMJ)). Please refer to White et al, 1986 for neuron naming conventions. Neuromuscular junctions in this file derive from actual reconstructions and do not include extrapolations (see Neuron Connectivity to Sensory Organs and Muscles below).