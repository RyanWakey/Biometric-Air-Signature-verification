# Dissertation For Swansea University that earned an 84% mark

Biometric Air Signature verification

The application uses 3D data (3D acceleration or 3D angular velocity) to classify air signatures.

With the DTW approach you compare the stored template of a specific individuals air signature with a newly inputted air signature.
With the FSL learning approach we use the database that stores a few air signatures for each individual to learn and classify from. 

I would recommend using your own dataset as the one provided contains a few individuals air signatures sampled at an incorrect sample rate, which 
causes sample bias effecting the quality of classification of the signatures. 
You could however just remove the incorrectly sampled air signatures from the database by removing the records that contain small amount of data compared to the others.

Not much fine tuning was done to the Model due to time restriction. 
I would recommend downloading Cuda if your system is applicable as it greatly speeds up both training and inference.
