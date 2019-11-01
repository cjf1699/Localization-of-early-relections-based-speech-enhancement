# Localization-of-early-relections-based-speech-enhancement
## Introdution
This project provides a speech enhancement scheme, which exploits the early reflections, which are strongly correlated with the direct sound, in an enclosed acoustic environments. The scheme contains 3 steps:
* First, the DOA(direction of arrival) of a few dominant reflections are needed.  
* Second, signal in the corresponding direction are extracted using some methods like beamforming. 
* Finally, the early reflections together with the direct sound would be fed into some nerual network to further enhance the speech.  
Up until now, only the first step is completed, so the code is all about localization of early reflections now. A deep residual nerual network is employed in the current work. A related paper is submitted to ICASSP2020.
## Description of some files
* config.py: global parameters
* net.py: the definition of nerual networks
* DataPreProcessor.py: the definition of dataset and code for data processing.
* handler.py: some functions like *peak detection*
* train.py: code for train and validation
