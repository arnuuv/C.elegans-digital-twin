**************************************************************************
**                                                                      **
**         README for "Albertson_Fig" directory (folder)                **
**                                                                      **
**                                                                      **
**                    CCeP (Cybernetic Caenorhabditis elegans Program), **
**                                                  December 28, 2003.  **
**                                                                      **
**************************************************************************


=================================
 INTRODUCTION
=================================
Files in this "Albertson_Fig" directory (folder) describe the synaptic
connectivity shown in the figures in the paper:

  Albertson, D. G. and Thomson, J. N. (1976).
  "The Pharynx of Caenorhabditis elegans",
   Phil. Trans. R. Soc. London B 275, pp.229-325.

In that paper, authors have studied the nervous system in the pharynx.
The pharynx is composed of 14 classes of 20 neurons.


=================================
 NOTICE
=================================
The way to construct this database is similar to one for White et al. (1986).
However the two database have some differences because the figures in the
two papers are different. In Albertson and Thomson (1976),

1. Mostly the class name is used for the partner neuron (the 6th column).
2. There is another synapse type "M" that represents a muscle synapse
   (the 5th column). In the original paper, a series of triangles stands
   for muscle synapses.
3. The index in the 7th column represents a variable synapse rather than
   a multiple one.


=================================
 FILES AND DIRECTORIES (FOLDERS)
=================================

 - [Albertson_Fig] -+- README_AF.txt
                    +- synapse_AF.txt
                    +- process_AF.pdf
                    |
                    +- [neurons] --- XXX_AF.txt


  o README_AF.txt:
      This file. ASCII text file that gives general information on files
      in this "Albertson_Fig" directory (folder).

  o synapse_AF.txt:
      ASCII text file in which all of the data files "XXX_AF.txt" in the
      "neurons" directory (folder) are included. For MI (a motor-interneuron),
      however, only "MI_upper_AF.txt" is included to make this file. There
      is no biological reason why we choose not "MI_lower_AF.txt" but
      "MI_upper_AF.txt" (see below for these two data files).

  o process_AF.pdf:
      PDF file that illustrates the correspondence between the "process"
      (branched morphology of neuron) and the index of process (2nd column
      in the data files "synapse_AF.txt" and "XXX_AF.txt").

  o XXX_AF.txt in the "neurons" directory (folder):
      ASCII text files that describe the synaptic connectivity shown in the
      figures in the original paper for each reference neuron. Here XXX
      denotes a name of reference neuron.

      In the original paper, MI is illustrated in two figures because MI is
      highly variable in the synaptic connectivity. We provide in 
      "MI_upper_AF.txt" and "MI_lower_AF.txt" data illustrated in the upper
      and the lower figures, respectively, in the original paper.


=================================
 FORMAT OF DATA
=================================
To exactly list the synaptic connectivity shown in the figures in the original
paper, we describe a synapse by the eight items which represent the followings.

  (1) name of reference neuron
  (2) the (branching) process of neuron
  (3) location of a synaptic contact (a dot illustrated in the original paper)
      on the process
  (4) which symbol of synapse type is attached to a synaptic contact?
  (5) a type of synaptic contact
  (6) (class) name of partner neuron
  (7) whether a synapse is variable or not?
  (8) comment on an ambiguous or questionable synaptic contact

See also "NOTE_WF.pdf" in the "White_Fig" directory (folder).
Corresponding to these eight items, the data files "synapse_AF.txt" and
"XXX_AF.txt" consist of eight columns.


[1st column] Name of reference neuron

[2nd column] Index of process:
  Each process of neuron is labeled with an alphabet. Refer to
  "process_AF.pdf" for the correspondence between the present index and
  the process illustrated in the original paper.

[3rd column] Serial number of synaptic contact:
  All the synaptic contacts and synapses onto muscles on a process are
  numbered. Consecutive numbers are used for a given reference neuron.
  From a combination of the 2nd column (the index of process) and the 3rd
  column (the present number), location of contacts on the process is
  known up to topology. If the location of more than one synapse are not
  distinguished, the same number is assigned in this column. 

[4th column] Serial number of synapse symbol:
  All the symbols of synapse type (arrows, equals and triangles in the
  original figures) are numbered. Consecutive numbers are used for a given
  reference neuron. The different number is assigned for every symbol of
  synapse type.

[5th column] Index of synapse type:
  Synapse type is illustrated with an arrow (a chemical synapse), an equal
  (a gap junction) or a triangle (a muscle synapse) in the original paper.
  In the case of chemical synapse, furthermore, "send" or "receive" is
  distinguished by direction of the arrow. We use five indices, "S", "R",
  "G", "M" and "-" to descrive the synapse type.

  "S": The reference neuron sends chemical synapse to its partner neuron.
  "R": The reference neuron receives chemical synapse from its partner neuron.
  "G": A gap junction connects the reference neuron with its partner neuron.
  "M": The synapse is a muscle synapse. 
  "-": A symbol of synapse type is missing.

[6th column] Class name of partner neuron or name of muscle cell:
  In the original paper, the partner neurons is indicated by the class
  name rather than the neuron name. Therefore this column is filled with
  class name unless the neuron name is given explicitly in the original
  figures. In the case of muscle synapse, a name of one muscle cell is usually
  denoted near a series of more than one triangle in the original paper. The
  name of muscle cell is assigned to a synapse which is illustrated by the
  nearest triangle to the muscle name and "-" is assigned to remaining
  synapses.

[7th column] Index for variable synapse:
  The authors of the original paper call the synapse variable when it is
  found on a single series only. The variable synapse is represented by
  parentheses in the original figures. This column is "v" when the connection
  is variable and is "-" when not.

  "v": variable synapse.
  "-": otherwise.

[8th column] Comment index:
  When descriptions in the original figures are questionable, we have commented
  in this column. We have classified the comments into 15 types, and distinguish
  them by the following indices.

     "-": Nothing to comment.
   "af1": No symbol of synapse type is attached in the original paper.
   "af2": Comment peculiar to I2R. "af1" and that it is possible that the
          synapse lies on the process B of I2L.
   "af3": Comment peculiar to I3. The symbol of synapse type is denoted at
          the outside of the bracket against the general rule of the original
          figure.
   "af4": There is an unprescribed semibracket.
   "af5": There is an ambiguity as to the partner(s) of the muscle synapse.
   "af6": Comment peculiar to M3L and M3R. The synapse is located at the 
          border of the process D and the cell body.
   "af7": Comment peculiar to M3R. It is possible that the partner of this 
          synapse is MI.
   "af8": There is(are) other type(s) of synapse between this muscle synapse
          and the location where the partner is indicated.
   "af9": Location of the synaptic contact is erroneous.
  "af10": "af5" and "af8".
  "af11": Comment peculiar to NSML. "af1" and that it is possible that the
          synapse lies on the process B of NSMR.
  "af12": Comment peculiar to NSMR. It is possible that the synapse lies on 
          the process C.
  "af13": Comment peculiar to NSMR. It is possible that the synapse lies on 
          the process B of NSML.
  "af14": The synapse data is taken from the upper figure of MI in the
          original paper, and is included in "synapse_AF.txt".
  "af15": The synapse data is taken from the lower figure of MI in the
          original paper, and is not included in "synapse_AF.txt".


=================================
 EXAMPLES OF DATA
=================================

(1) Two rows where the 3rd column is the same and the 7th column is "-":
  I2L  B 5 5 S I6  - -
  I2L  B 5 6 S NSM - -   (in I2L_AF.txt) 
represents "-> I6" and "-> NSM" in the same location on the process.

(2) Two rows where the 3rd column is the same and the 7th column is "v":
  M1  C 26 29 R M4 v -
  M1  C 26 30 G I4 v -   (in M1_AF.txt) 
represents "(<- M4, = I4)".

(3) The more complicated case of four rows where the 3rd column is the same:
  MCL  B 2 2 R I1 - -
  MCL  B 2 3 G M2 - -
  MCL  B 2 4 R I2 v -
  MCL  B 2 5 R M1 v -     (in MCL_AF.txt)
represents "<- I1", "= M2", "(<- I2)" and "(<- M1)" in the same location
on the process.

(4) A pair of rows where the 4th column is the same:
  I1L  B 9 9 S M2 - -
  I1L  B 9 9 S M3 - -    (in I1L_AF.txt)
represents "-> M2, M3".

(5) The 5th column contains two characters:
  IL1  B 6 6 S/G I5 - -  (in I1L_AF.txt)
represents "-> or = I5"

