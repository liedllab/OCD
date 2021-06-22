# Introduction

OCD.py is a command-line tool for the calculation of immunoglobulin inter-domain orientations. 
The OCD tool automatically creates a reasonable coordinate system for the characterization of inter-domain orientations based on a user-provided reference structure. 
While OCD.py is free to use, please cite the following paper if you use it in your scientific work:
https://www.biorxiv.org/content/10.1101/2021.03.15.435379v1

# Installation

OCD.py runs on Mac OS and Linux. It is supported for Python versions 3.6-3.8 

We recommend using conda (https://docs.anaconda.com/anaconda/install/) to create a fresh environment :
> conda create -n OCDpy3.7 python=3.7  
> conda activate OCDpy3.7  

Install the tool with pip from inside the OCD folder:
> cd /PATH_TO_OCD/OCD  
> pip install .  

If you want to run OCD on Python 3.8, first install the ambertools package using conda:
> conda install -c conda-forge ambertools=20

Alternatively, install the dependencies by yourself. Then add OCD.py to your PATH variable.

# Dependencies
OCD.py depends on the following libraries. Find installation instructions included in the provided links.
- pytraj https://amber-md.github.io/pytraj/
- mdtraj https://mdtraj.org
- pandas https://pandas.pydata.org/
- matplotlib https://matplotlib.org/
- numpy https://numpy.org/

# Usage

Run OCD.py from the console:
> OCD [Arguments]

To get a summary of all available arguments, type:
>OCD -h

The following arguments are necessary:
> -i <Input>        
> Provide a file path to a structural data file (structure or simulation) here. 
>
> -t <Topology>     
> Provide a file path to a topology file here. This is only necessary if a MD trajectory was input. 
>
> -A / -mask_A <Mask_A>     and     -B / -mask_B <Mask_B>  
> Provide an atom selection for domain A and B according to the AMBER syntax: https://amberhub.chpc.utah.edu/atom-mask-selection-syntax/. Note that both are required to run OCD.py!

The calculation will be run for domains A and B as specified thorugh the mask selection.

The following argument is highly recommended, though not strictly required:
> -r <Reference>        
> Provide a file path to a structural data file (structure or simulation) here. This will be used as the reference structure for the calculation. Make sure the residues in the reference are annotated the same as the input data. If this is not provided, the first frame of the input structural data will be used as reference instead. 

The following arguments are optional:
> -use <First> <Last> <n>      
If a MD simulation is analyzed, this argument specifies which frames will be considered. Only every <n>-th frame between frames <First> and <Last> will be considered.

> -o / -output <Output>     
Set a name to be used for all output files. Output files will be called OCD_<Output>.*

> -\-pdb     
> A pdb file of the first frame in the input data will be output. This is helpful to make sure the residues are numbered the same between the reference and input structures.
>
> -\-vmd     
> A tcl script containing the data of the coordinate system is output. Use this for MD trajectory analysis. Additionally, the full input trajectory is aligned on the first frame and output as well.
>
> -\-pymol   
> A pym script is generated. Use this to visiualize the generated coordinate system with PyMOL.
>
> -\-plot [hist and/or time]     
> Set this to plot the generated data for MD simulation trajectories. Either histogramms, time series or a plot containing both can be generated.
>
> -\-lim_AB <Lower> <Upper>      
> Set x-axis limits in the generated plots for the measure AB. Plots will only show the measures between the two provided bounds. Replace AB with AC1, AC2, BC1, BC2 or dc to similarly set the bounds for the opther plots.
>
