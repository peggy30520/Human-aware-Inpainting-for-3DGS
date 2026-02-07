# Targeted-pruning LightGaussian

The codebase is based on  [LightGaussian](https://github.com/VITA-Group/LightGaussian), modified part is shown as below:
* Adding mask input (image & bounding box)
* Training loss excluded mask region
* Prune the unnecessary splats inside bounding box 
