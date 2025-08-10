# findHopper🦗 

This is a forked copy of the [xsrp](https://github.com/egrinstein/xsrp) repository. Some of the testing functions from the original libary were removed.

It includes some additional code to do source localization for microphone array recordings of grasshoppers and crickets (see also [thunderhopper](https://github.com/bendalab/thunderhopper) repository). 

<iframe width="560" height="315" src="https://www.youtube.com/embed/ajGM4t9v8g4?si=NyuYu2HintGBqQLJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


## Installation

I would recommend installing a [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) virtual environment using the provided `environment.yml` file. This will install all the required dependencies. Change directory to the ```findhopper``` folder and run the following commands:

```
1. conda env create -f environment.yml
2. conda activate findhopper
```


## Testing with sample data

Download sample data here: https://whale.am28.uni-tuebingen.de/nextcloud/index.php/s/Jg9iyR397peCE96


## References

Grinstein, E. et al. (2024) ‘Steered Response Power for Sound Source Localization: A Tutorial Review’. arXiv: https://doi.org/10.48550/arXiv.2405.02991.

bendalab/thunderhopper: Model of the auditory pathway of grasshopper: https://github.com/bendalab/thunderhopper


