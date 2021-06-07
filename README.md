**Copyright &copy; 2021, Sebastian Andreß**\
All rights reserved. Please find the license [here](https://github.com/sebastianandress/Slicer-AffinityClusterRegistration/blob/master/LICENSE.md).

Please cite the corresponding paper when using this filter for publications:

    @article{HighAccuracySurfaceZones,
        author      = {Andreß, Sebastian and Achilles, Felix and Bischoff, Jonathan and Calvalcanti Kußmaul, Adrian and Böcker, Wolfgang and Weidert},
        title       = {A Method for Finding High Accuracy Surface Zones on 3D Printed Bone Models},
        journal     = {Computers in Biology and Medicine},
        publisher   = {Elsevier},
        date        = {2021-09-01},
    }


![Header](/Resources/header.png)

# Similarity Subgroups Validation Method

## Introduction
This tool is intended to find different surface error types. Unless traditional methods, it tries to find correlating subgroups of vertices, that show high accuracy (within this subgroup) and uses those for registration.

It tries to close the gap between rigid and deformable registration and validation methods. As described in the paper mentioned above, this method is tested for four different types of model errors.

The main purpose of this module is to find high accuracy surface zones (above a certain area size), that show low deviation (within a certain threshold) when taken for themselves, regardless of whether they deviate significantly from the entire model.

![Screenshot](/Resources/screenshot1.png)

## Description
1. Mark at least 3 landmarks (better >5) on both the source and target model in the same order for the pre-registration.
2. Select reasonable parameters, they are described in the paper mentioned above.
3. Apply the algorithm.
4. Results are stored as scalars in the source model, you can view those in the "Models" Module setting scalars to visible and selecting the scalar accordingly.

## How to install
The Extension is available in the [Extension Manager](http://slicer.kitware.com/midas3/slicerappstore/extension/view?extensionId=330842) for Slicer Versions greater than 4.11.
To install it manually, please follow the description on the official [3D Slicer page](https://www.slicer.org/wiki/Documentation/Nightly/Developers/FAQ/Extensions). 
