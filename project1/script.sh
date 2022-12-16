#!/bin/bash

# chmod u+x script.sh

for i in {1..1}
do
   python3 project1-cnn.py
   echo " "
   echo "*******************************************"
   echo " "
   
   python3 project1-cnn-siamese.py
   echo " "
   echo "*******************************************"
   echo " "
    
   python3 project1-siamese.py
   echo " "
   echo "*******************************************"
   echo " "
   
   python3 project1-siamese-nn.py 
   echo " "
   echo "*******************************************"
   echo " "
   
   python3 project1-siamese-all.py  # predict 100% acc.
   echo " "
   echo "*******************************************"
   echo " "
   
   python3 project1-siamese-nn-all.py # predict 100% acc.
done


