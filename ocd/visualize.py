#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vhoer

Functions used by OCD.py for visualisations.
"""

import matplotlib.pyplot as plt
import numpy as np
import textwrap

    
def binrange(min_data, max_data, binsize):
    '''
    Generates array of bin positions starting from min_data and ranging to
    max_data using a stepsize of binsize.
    Parameters:
    -----
    Requires:
        min_data; lowest value in the data to be binned as float
        max_data; highest value in the data to be binned as float
        binsize; size of a single bin as float
        
    Returns:
        a numpy array; contains all binpositions 
    '''
    return np.arange(min_data, max_data+binsize, binsize)    

def angle_plots(x_data, 
                y_time, 
                x_comp=[], 
                xlabel=None, 
                title=None, 
                wghts = None, 
                bin_dims=None, 
                rm_frac = 100, 
                norm = False, 
                hist1c = 'grey', 
                hist2c = 'blue', 
                linec = 'darkblue', 
                xlim = None,
                plot_type = ['hist', 'time']):
    '''
    Creates OCD Angle plots. x_data is one ABangle like measurement data set
    (eg dc) as array with size N, x_comp the same but from the 
    ABangle comparison set and y_time is the time data set (array of size N). 
    Title is set with title. wghts is an array of size N used for 
    reweighting x_data. bin_dims give the bin dimension in the 
    following format: (start, end, stepsize). rm_frac is used for 
    running average: Number of frames to average is calculated 
    by len(y_time)/rm_frac.
    
    Parameters:
    -----
    Requires:
        xdata;
        y_time;
        x_comp;
        x_label;
        title;
        wghts;
        bin_dims;
        rm_frac;
        norm;
        hist1c;
        hist2c;
        linec;
        xlim; 
        plot_type; list containing strings which tell the function what to plot: 
            'hist' will plot histograms, 
            'time' will plot a scatter plot of the data.
            Providing both plots a histogram on top of the time series plot.
    
    Returns:
        f, a pyplot figure element
    '''
    plt.rcParams.update({'font.size': 26})

    ### Unpack bin dims ###
    min_data, max_data, stepsize = bin_dims[0], \
                                   bin_dims[1], \
                                   bin_dims[2]

    ### Calculate Weights for normalization
    # norm so that sum of all bins equals 1. Density = True. 
    # Occurence in percent.
    if norm == True: 
        wghts = np.ones_like(x_data)*100/float(len(x_data))
        wghts_comp = np.divide(np.ones_like(x_comp)*100, float(len(x_comp)))
    else:
        wghts_comp = None

    ### Make Subplots. If only_hist is True, 
    #   make only the histograms without the timeline
    for ix in plot_type:
        if ix not in ['hist','time']:
            print('Angle_plots: \t Wrong arguments in plot_type.' 
                  'Only "hist" or "time" are allowed.' 
                  'Please fix and try again.')
            return
    
    if len(plot_type) == 1:
        f, ax1 = plt.subplots(figsize=(8,8))
    else:
        f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8,16))
        f.subplots_adjust(hspace=0)

    ### Plotting ###
    plotted_hist, plotted_time = False, False
    for ax in f.axes:
        if 'hist' in plot_type and plotted_hist == False:
            if len(x_comp) != 0:  
                n_comp, bins_comp, patches_comp = ax.hist(x_comp, 
                                                          bins=binrange(min_data, max_data, stepsize), 
                                                          density=False, 
                                                          histtype='barstacked', 
                                                          stacked=True, 
                                                          alpha=0.2, 
                                                          color=hist1c,
                                                          edgecolor='black', 
                                                          weights=wghts_comp, 
                                                          label='Comparison data set')
                                                          
                n_comp_outline, bins_comp_outline, patches_comp_outline = ax.hist(x_comp,
                                                                                   bins=binrange(min_data, max_data, stepsize), 
                                                                                   density=False,
                                                                                   histtype='step', 
                                                                                   stacked=True, 
                                                                                   color=hist1c, 
                                                                                   weights=wghts_comp, 
                                                                                   label='Comparison data set')
                                                                                   
            n_data, bins_data, patches_data = ax.hist(x_data, 
                                                      bins=binrange(min_data, max_data, stepsize), 
                                                      density=False, 
                                                      histtype='barstacked', 
                                                      stacked=True, 
                                                      alpha=0.5, 
                                                      color=hist2c, 
                                                      edgecolor='black', 
                                                      weights = wghts, 
                                                      label= 'Calculated data set')
                                                      
            n_data_outline, bins_data_outline, patches_data_outline = ax.hist(x_data, 
                                                                              bins=binrange(min_data, max_data, stepsize), 
                                                                              density=False, 
                                                                              histtype='step', 
                                                                              stacked=True, 
                                                                              color=hist2c, 
                                                                              weights=wghts, 
                                                                              label='Calculated data set')
            ax.set_ylabel('Occurence / %')
            ax.set_ylim(0, 25)
            plotted_hist = True
            continue
        
        if 'time' in plot_type and plotted_time == False:
            ax.scatter(x_data,y_time, label = 'Frames', s=1, color=hist2c)
            N = int(len(y_time)/rm_frac)
            x_avg = x_data.rolling(N).mean()
            ax.plot(np.array(x_avg),np.array(y_time), 
                    label = 'Rolling mean: n = {!s}'.format(N), 
                    color = linec)
            ax.set_ylabel('Time / ns')

    ### Formatting ###
    f.axes[0].set_title(title)
    f.axes[-1].set_xlabel(xlabel)
    if xlim == None:
        plt.xlim(np.floor(min_data), np.ceil(max_data))
    else:
        plt.xlim(xlim[0],xlim[1])
       
    ### Return figure ###
    return f


### Pymol specific code starts here
                 
def pymol_init():
    '''
    Writes a string to set up the groups in PyMol used in pymol_moi
    
    Parameters:
    ----
    Requires nothing
    
    Returns a string of the necessary PyMol commands
    
    '''
    str_o = "cmd.group(name='pdbs', action='open')\n"
                "cmd.group(name='pseudoatoms', action='open')\n"
                "cmd.group(name='coms', action='open')\n"
                "cmd.group(name='distances', action='open')\n"
    return str_o

def pymol_settings(masks = []):
    '''
    Changes visualization in pymol, add all settings which need to be set only once here
    
    Parameters:
    -----
    Requires nothing
    
    Optional arguments:
        masks, a list of masks. Each entry should contain two str entries: a pymol maskstring and a pymol colorstring
    
     Returns a string of the necessary PyMol commands
    '''

    str_o = ("cmd.set('antialias_shader',2)\n"
            "cmd.set('dash_gap', 0)\n"
            "cmd.set('dash_radius', 0.5)\n"
            "cmd.set('cartoon_gap_cutoff', 0)\n"
            "cmd.set('cartoon_transparency', 0.2)\n"
            "cmd.set('cartoon_tube_radius',0.2)\n"
            "cmd.set('cartoon_fancy_helices',1)\n"
            "cmd.set('cartoon_cylindrical_helices',0)\n"
            "cmd.set('cartoon_highlight_color', -1)\n"
            "cmd.set('ribbon_radius',0.2)\n"
            "cmd.set('cartoon_flat_sheets',1)\n"
            "cmd.set('cartoon_smooth_loops',0)\n"
            "cmd.set('antialias', 1)\n"
            "cmd.set('ambient', 0.15)\n"
            "cmd.set('direct',1)\n"
            "cmd.set('specular', 0)\n"
            "cmd.set('ray_trace_mode',0)\n"
            "cmd.set('ray_shadows',0)\n"
            "cmd.set('ray_opaque_background',0)\n"
            "cmd.hide('labels')\ncmd.hide('lines')\n"
            "cmd.hide('everything','resn NME')\n"
            "cmd.color('grey90', 'pdbs')\n"
            "cmd.bg_color('white')\n"
            )
    return str_o    
 
    if len(masks) > 0:
        for mask in masks:
            str_o += "cmd.color({}, {})\n".format(mask[0], mask[1])  

            
def pymol_load(file, obj):
    '''
    Loads file into PyMol as obj.
    
    Parameters:
    ----
    Requires:
        file, a pdb file path or similar which is loadable into PyMol
        obj, a str to name the loaded object in PyMol
           
     Returns a string of the necessary PyMol commands
    
    '''

    str_o = ("cmd.load('{}','{}')\n"
             "cmd.group(name='PDBs', "
             "members='{}', action='add')\n"
            ).format(file, obj, obj)
    return str_o


def pymol_draw_vectors(vectors=[], distances=[]):
    '''
    Adds vector and distance objects to PyMOL
    Currently doesn't support different frames/states.
    Distances are drawn as thin cylinders, vectors are cylinders 
    with an added coneshape (arrows) pointing to the end coordinates.
    The colors and shapes can be changed in the highlighted section of
    this function by changing variables 
    'colorlist', 'scale', 'cone_scale' and 'middle_scale'.
    
    Parameters:
    -----
    Required:
        -
        
    Optional arguments:
        vectors; a list of shape (nr_vectors, 3) containing any 
            amount of vectors. Each vector is composed of three entries: 
            start coordinates ([x,y,z]), end coordinates ([x,y,z]) 
            and an int giving the color of the vector as entry in 
            variable 'color_tuples')
            Example containing two vectors: 
            [ [[0,0,0],[1,0,0],'black'], [[0,0,0],[0,1,0], 'tv_red'] ]
        distances; same as vectors.
    
    Returns:
        A string containing the necessary PyMOL commands
        
    '''
    if len(vectors) == 0 and len(distances) == 0:
        return
    script_str=''
    #### CHANGE DRAWING SETTINGS HERE
    colorlist = ['tv_red', 'tv_green', 'tv_blue', 'yellow']   
    color_tuples = [[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.3, 0.3, 1.0], [1.0, 1.0, 0.0]] #tv_red, tv_green, tv_blue, yellow
    scale = 10                                     
    r = 0.4                                         
    cone_scale = 2                                
    middle_scale = 0.75                              
    ####
    
    cone_r = r * cone_scale
   
    for i,vec in enumerate(vectors):
        color = vec.pop()
        vec = np.array(vec)
        vector = (vec[1] - vec[0]) * scale    
        x1, y1, z1 = vec[0]
        x2, y2, z2 = vec[0] + vector * middle_scale 
        x3, y3, z3 = vec[0] + vector
        obj = [9.0, x1, y1, z1, x2, y2, z2, r, 
               *color_tuples[color], *color_tuples[color], 
               27.0, x2, y2, z2, x3, y3, z3, cone_r, 0.0, 
               *color_tuples[color], *color_tuples[color], 1.0, 1.0]

        script_str += ("obj = {}"
                       "\ncmd.load_cgo(obj, 'Vector_{}')\n"
                      ).format(str(obj),i)  
    for i,vec in enumerate(distances):
        color = vec.pop()
        vec = np.array(vec)      
        x1, y1, z1 = vec[0]
        x2, y2, z2 = vec[1]
        obj = [9.0, x1, y1, z1, x2, y2, z2, r, 
               *color_tuples[color], *color_tuples[color]]
        script_str += ("obj = {}"
                       "\ncmd.load_cgo(obj, 'Distance_{}')\n"
                      ).format(str(obj),i)  
    
    return script_str
        
#### VMD specific code starts here

def vmd_script(traj, top, use, vectors, distances=[], 
               out='vmd', delete_arrows = True):
    '''
    Generates a vmd script to visualize coordinate system as defined by
    eigenvectors of the inertia tensor. Can display any number of vectors 
    and distances per frame. Vectors are drawn as arrows 
    (thin cylinder + cone), distances are drawn as thin cylinders.
    Be aware that a trajectory file of the trajectory used for the calculation
    should probably be written out beforehand, to ensure that the trajectory 
    visualized and the trajectory for which the calculations were carried out 
    are one and the same. After creation, the script can be visualized by 
    calling 'vmd -e "out"_vmd.tcl', 
    where "out" is the string given as argument "out".
    The script can be accessed and changed in variable "script".
    
    Parameters:
    -----
    Required:
        traj, a string containing the path to the trajectory file to display
        top, a string containing the path to the topology file of traj
        use, a tuple (start, stop, step) containing the first frame, 
             last frame and stepsize of the trajectory to be displayed. If 
             trajectory file was written out beforehand this should be (0, -1, 1).
             vectors, a numpy array of shape (nr_frames, nr_vectors, 3). 
             Each entry is a frame containing all vectors to be drawn for 
             this frame. Each vector contains three entries: a starting point as a 
             list of x y z coordinates, an end point as a list of x y z coordinates
             and an int giving the color of this vector as index of the colorlist
             which is defined in the script variable.
        out, a string containing a name for the output. The resulting script
             will be output to ./"out"_cmd.tcl
   
    Additional:
        distances, a numpy array of shape (nr_frames, nr_distances, 3). 
                   Each entry represents a frame containing any number 
                   of distances. Each distance consists of a start point 
                   (numpy array with shape (3) of xyz coordinates), 
                   an end point (same as start point) and a color index 
                   (int, giving the index of the color to nbe used as 
                   defined in the scripts colorlist)
        delete_arrows, a boolean (default is True) flag. When set to False, 
                       the drawn arrows won't be replaced each frame and 
                       will persist. This allows the visualization of all
                       vectors in one frame, though each frame has to be
                       shown in VMD once.
    
    Returns:
        -
    
    A file will be written to ./"out"_cmd.tcl
    '''
    draw_distances = False
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}, threshold=np.inf)
    
    
    tcl_string = "set vec_data " + str(vectors).replace('[','[list ').replace('\n','').replace("'","").replace(',','') + '\n\n'  
                                                                
    if len(distances) != 0:
        draw_distances = True
        tcl_string += "set dist_data " + str(distances).replace('[','[list ').replace('\n','').replace("'","").replace(',','') + '\n\n'
                                                                                                          
    header = ('mol new {} type parm7 waitfor all\nmol addfile {} type netcdf' 
             'first {} last {} step {} waitfor all\nmol delrep 0 top\nmol' 
             'representation NewCartoon 0.300000 10.000000 4.100000 0\nmol' 
             'addrep top\n'.format(top, traj, use[0], use[1], use[2])
             )
    ### This doesn't need the double {{ }} as it's not affected by format
    distance_script = """\
                    global dist_data
                    set distances [lindex $dist_data $frame]
                    foreach distance $distances {
                            set distance_color [lindex $colorlist [lindex $distance 2]]
                            draw color $distance_color
                            graphics [molinfo top] cylinder [lindex $distance 0] [lindex $distance 1] radius 0.6
                            }"""
                   
    
    ### This needs the double {{ }} because it's affected by format()
    script = """\
                set colorlist [list red green blue black]
                #--------------------------------------------------------#

                proc vmd_draw_arrow {{mol start end color}} {{
                    
                    ###### CHANGE ARROW SHAPE HERE #####
                    set cone_scale 1.5
                    set radius 0.6
                    set scale 10
                    set middle_scale 0.8
                    
                    ##### ARROW DRAWING STARTS HERE #####
                    set scaled_vec [vecscale $scale [vecsub $end $start]]
                    set scaled_end [vecadd $start $scaled_vec]
                    draw color $color
                    set radius_cone [expr {{$radius * $cone_scale}}]
                    set middle [vecadd $start [vecscale $middle_scale $scaled_vec]]
                    graphics $mol cylinder $start $middle radius $radius
                    graphics $mol cone $middle $scaled_end radius $radius_cone
                }}

                #--------------------------------------------------------#

                proc update_frame {{}} {{
                    # Traces change to global frame number, executes draw_frame on change
                    global vmd_frame
                    trace variable vmd_frame([molinfo top]) w draw_frame
                }}

                #--------------------------------------------------------#

                proc draw_frame {{a b c}} {{
                    global vec_data
                    global vmd_frame 
                    global colorlist
                    draw delete all
                    set frame $vmd_frame([molinfo top])
                    set vectors [lindex $vec_data $frame]
                    foreach vector $vectors {{
                            set vector_color [lindex $colorlist [lindex $vector 2]]
                            draw arrow [lindex $vector 0] [lindex $vector 1] $vector_color         
                            }}
                    {distance_drawing}
                    
                    }}


                update_frame
                animate goto start
            """.format(distance_drawing=distance_script if draw_distances == True else '' )
                
    if delete_arrows == False:
        script = script.replace('draw delete all\n','')
    with open('./{}_vmd.tcl'.format(out), 'w+') as f:
        f.write(header)
        f.write(tcl_string)
        f.write(textwrap.dedent(script))     
    np.set_printoptions()
