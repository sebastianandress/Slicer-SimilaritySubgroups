cmake_minimum_required(VERSION 3.13.4)

project(SimilaritySubgroups)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/sebastianandress/Slicer-SimilaritySubgroups")
set(EXTENSION_CATEGORY "Registration")
set(EXTENSION_CONTRIBUTORS "Sebastian Andreß (LMU Munich)")
set(EXTENSION_DESCRIPTION "This is an example of a simple extension")
set(EXTENSION_ICONURL "https//www.github.com/sebastianandress/Slicer-SimilaritySubgroups/SimilaritySubgroups/Resources/SimilaritySubgroups.png")
set(EXTENSION_SCREENSHOTURLS "https//www.github.com/sebastianandress/Slicer-SimilaritySubgroups/Resources/screenshot1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SimilaritySubgroups)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
