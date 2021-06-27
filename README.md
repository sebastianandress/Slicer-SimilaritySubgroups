**Copyright &copy; 2021, Sebastian Andreß**\
All rights reserved. Please find the license [here](https://github.com/sebastianandress/Slicer-AffinityClusterRegistration/blob/master/LICENSE.md).

Please cite the corresponding paper when using this filter for publications:

    @article{HighAccuracySurfaceZones,
        author      = {Andreß, Sebastian and Achilles, Felix and Bischoff, Jonathan and Calvalcanti Kußmaul, Adrian and Böcker, Wolfgang and Weidert},
        title       = {A Method for Finding High Accuracy Surface Zones on 3D Printed Bone Models},
        journal     = {Computers in Biology and Medicine},
        publisher   = {Elsevier},
        year        = {2021},
        doi         = {https://doi.org/10.1016/j.compbiomed.2021.104590}
    }


![Header](/Resources/header.png)

# Similarity Subgroups Model Validation Method

## Introduction
This tool is intended to find different surface error types. Unless traditional methods, it tries to find correlating subgroups of vertices, that show high accuracy (within this subgroup) and uses those for registration.

It tries to close the gap between rigid and deformable registration and validation methods. As described in the paper mentioned above, this method is tested for four different types of model errors.

The main purpose of this module is to find high accuracy surface zones (above a certain area size), that show low deviation (within a certain threshold) when taken for themselves, regardless of whether they deviate significantly from the entire model.

![Screenshot](/Resources/screenshot1.png)

## Example

![ExampleOutput](/Resources/output.pdf)

Two boxes are compared with each other. Both are identical, but one is broken in the middle into two pieces. The algorithm recognizes this fact and registers both halves of the source box with the target box, thus creating a "Similarity Subgroup" for each half. For each subgroup considered as such, the deviation is zero.

## Description
1. If you want to use the pre-registration, mark at least 3 landmarks (better >5) on both the source and target model in the same order.
2. Select reasonable parameters, they are described in the paper mentioned above.
3. Apply the algorithm.
4. Results are stored as scalars in the source model, you can view those in the "Models" Module setting scalars to visible and selecting the scalar accordingly.

## How to install
The Extension will be available in the [Extension Manager](http://slicer.kitware.com/midas3/slicerappstore/extension/view?extensionId=330842) soon.
To install it manually, please follow the description on the official [3D Slicer page](https://www.slicer.org/wiki/Documentation/Nightly/Developers/FAQ/Extensions). 
