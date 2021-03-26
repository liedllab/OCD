#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vhoer

Current version of the OCD script for the characterization of protein
inter-domain interface orientations. 
Generates a coordinate system on the fly based on the first frame of 
the trajectory or a reference structure provided by the user. 
Then this coordinate system and the reference are used to calculate
 six orientational measures.
"""

from ocd import visualize as vis
from ocd import calculation as calc

import pytraj as pt
import numpy as np
import argparse
import sys
import os 
import time
import math
import pandas as pd
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import matplotlib.pyplot as plt
from mdtraj.geometry import alignment as al
from mdtraj.geometry.alignment import compute_transformation

xvec = [1, 0, 0]
yvec = [0, 1, 0]
zvec = [0, 0, 1]
start= [0,0,0]
simtime = 1000

def argumentParser():
    '''
    Parses user input from commandline. Help for each command 
    is given in its help section.
    
    Parameters:
    -----
    Requires:
        -
    
    Returns:
        argspace, a namespace containing all the arguments provided by the user
    '''
    parser = argparse.ArgumentParser(
                        description='Test for allignment based dynamic orientation.')

    parser.add_argument('-i', 
                        '-input', 
                        dest='input', 
                        required=True,  
                        help='Input coordinate file here')
    parser.add_argument('-t', 
                        '-topology', 
                        dest='top', 
                        required=False, 
                        default='', 
                        help='Input topology file here')
    parser.add_argument('-use', 
                        type=int, 
                        dest='use', 
                        default=None, 
                        metavar='', 
                        nargs=3, 
                        help='Framemask for creating pdbs from the trajectory.'
                            'Needs three arguments: FirstFrame LastFrame Stride.'
                            'Default: 0 -1 1')
    parser.add_argument('-mask_A',
                        '-A',
                         default="", 
                         dest='A',
                         help='Write the atom mask for domain A here')
    parser.add_argument('-mask_B',
                        '-B',
                        default="",
                        dest='B',
                        help='Write the atom mask for domain B here')
    parser.add_argument('-o',
                        '-output', 
                        default="Run_"+time.strftime("%Y-%m-%d_%H-%M-%S"), 
                        dest='output', 
                        help='Name to be prepended for all files output by this script')
    parser.add_argument('--align',
                        default=False,
                        dest='align',
                        action='store_true', 
                        help='Set this flag to align trajectory beforehand, eg. for visualization purposes')
    parser.add_argument('-reference',
                        '-r',  
                        dest='refstruc', 
                        default=None, 
                        help='Give path to structure file to be used as reference structure.' 
                            'The same mask as for the input is applied to this as well.')
    parser.add_argument('--vmd', 
                        default=False, 
                        dest='vmd', 
                        action='store_true', 
                        help='Set this flag to output a vmd script and a trajectory to be used for visualizing the ABangle vectors')
    parser.add_argument('--pymol', 
                        default=False, 
                        dest='pymol', 
                        action='store_true', 
                        help='Set this flag to show the first frame with the coordinate system in pymol')
    parser.add_argument('--pdb', 
                        default=False, 
                        dest='pdb', 
                        action='store_true', 
                        help='Set this flag to save the first frame as pdb file')
    parser.add_argument('--plot', 
                        default=False, 
                        dest='plot', 
                        action='store_true', 
                        help='Set this flag to output plots for MD data analyses, showing the histogram and change over time of the ABangles')
    parser.add_argument('--plot_type', 
                        default=['hist','time'], 
                        dest='plot_type', 
                        nargs='+', 
                        help='Specify which kind of plot should be generated. Available options are "hist", "time" or both.')
    parser.add_argument('--lim_AB', 
                        type=float, 
                        dest='lim_AB', 
                        default=None, 
                        metavar='', 
                        nargs=2, 
                        help='Set x-axis limits for the AB plot. Takes two arguments <LowerLim> and <UpperLim> as floats')
    parser.add_argument('--lim_AC1', 
                        type=float, 
                        dest='lim_AC1', 
                        default=None, 
                        metavar='', 
                        nargs=2, 
                        help='Set x-axis limits for the AC1 plot. Takes two arguments <LowerLim> and <UpperLim> as floats')
    parser.add_argument('--lim_AC2', 
                        type=float, 
                        dest='lim_AC2', 
                        default=None, 
                        metavar='', 
                        nargs=2, 
                        help='Set x-axis limits for the AC2 plot. Takes two arguments <LowerLim> and <UpperLim> as floats')
    parser.add_argument('--lim_BC1', 
                        type=float, 
                        dest='lim_BC1', 
                        default=None, 
                        metavar='', 
                        nargs=2, 
                        help='Set x-axis limits for the BC1 plot. Takes two arguments <LowerLim> and <UpperLim> as floats')
    parser.add_argument('--lim_BC2', 
                        type=float, 
                        dest='lim_BC2', 
                        default=None, 
                        metavar='', 
                        nargs=2, 
                        help='Set x-axis limits for the BC2 plot. Takes two arguments <LowerLim> and <UpperLim> as floats')
    parser.add_argument('--lim_dC', 
                        type=float, 
                        dest='lim_DC', 
                        default=None, 
                        metavar='', 
                        nargs=2, 
                        help='Set x-axis limits for the DC plot. Takes two arguments <LowerLim> and <UpperLim> as floats')
    parser.add_argument('--simtime', 
                        type=int, 
                        dest='simtime', 
                        default=1000, 
                        metavar='', 
                        help='Sets the simulated time in ns. Default is 1000 ns.')

    if len(sys.argv) == 1: #print help if no arguments are provided
        parser.print_help()
        calc.sysexit(1)
        
    return parser.parse_args()

def main():
    
    print('\nWelcome to OCD.py!\n')
    
    ### Load arguments from parser
    args = argumentParser()
    
    ### If no topology was provided, assume that the input 
    ### can be used as its own topology (eg pdb files)
    if args.top == '':
        print('No topology was provided.' 
              'The input file will be used as its own topology.')
        args.top = args.input

    ### Check if files exist
    for file in [args.input, args.top]:
        if not os.path.exists(file): 
            print('File {!s} doesn\'t exist.' 
                  'Please check the filepath or add the file.\n'.format(file))
            calc.sysexit(1)
    
    ### Format input arguments and generate some variables based on the user-provided arguments
    if args.A == "" and args.B == "":
        print( ("No atom selections were found. Please use -mask_A / -A and "
               "-mask_B / -B to select the domains you want to calculate "
               "the inter-domain orientation for. \nAtom selections are input "
               "as AMBER atom mask strings, see "
               "https://amberhub.chpc.utah.edu/atom-mask-selection-syntax/"
             ) )
        sysexit(0)
        
    if args.A != "" and args.B == "":
        print( ("WARNING: Only found an atom selection for domain A. Assuming "
                "domain B is everything not found in the domain A selection. "
                "If that's not correct, run OCD again and explicitly define "
                " domain B with the -B / -mask_B argument."
             )  )
        args.B = "!({})".format(args.A)
        
    if args.A == "" and args.B != "":
        print( ("WARNING: Only found an atom selection for domain B. Assuming "
                "domain A is everything not found in the domain B selection. "
                "If that's not correct, run OCD again and explicitly define "
                " domain A explicitly with the -A / -mask_A argument."
             )  )
        args.A = "!({})".format(args.B)
        
            
    
    
    
    # args.use needs to be a tuple for use in pytraj. 
    # If no argument was given, use every frame of the whole trajectory. 
    # Throws an error if input is not three arguments
    if args.use is None:
        args.use = (0,-1, 1)
    elif len(args.use) == 3:
        args.use = (args.use[0], args.use[1], args.use[2])
    else:
        print('Wrong number of arguments in -use.' 
        'Usage: -use <FirstFrame> <LastFrame> <Stride>')
        calc.sysexit(1)
    
    trajname = os.path.basename(args.input)
    
    ### Load trajectory
    print('Loading data.')
    traj_all = pt.iterload(args.input, args.top,frame_slice=args.use)
    
    #Align trajectory on first frame before running the calculations 
    if args.align or args.vmd: 
        traj_all = traj_all.superpose()
    
    ### Apply atom mask to trajectory, split the domains
    stripped_domains = [traj_all.strip('!('+args.A+')'), 
                        traj_all.strip('!('+args.B+')')]
    
    ### Get reference coordinates from reference structure or first frame, 
    #   if no reference structure was provided-
    #   Assuming the reference COM to be [0,0,0] is only valid if 
    #   the standard_orientation function is used, 
    #   so we need to reassign when args.abangleref == True
    ref_com_A = np.array([0,0,0])
    ref_com_B = np.array([0,0,0])   
    
    if args.refstruc != None:
        
        ref_traj = pt.iterload(args.refstruc)
        stripped_refs = [ref_traj.strip('!('+args.A+')'), 
                         ref_traj.strip('!('+args.B+')')]
        
        ### If residues are missing, exclude them from the 
        #   alignment with new_mask 
        #   - necessary because alignment would break if 
        #   the number of atoms supplied is different
        new_mask_A, new_mask_B = calc.new_masks(stripped_domains, stripped_refs)
        new_masks = [new_mask_A, new_mask_B]

        ## Strip domains down for alignment, if 
        #  reference structure has fewer residues. 
        #  Same thing happens for the residue structure 
        #  in the function standard_orientation.
        for dix, domain in enumerate(stripped_domains):
            try:
                if new_masks[dix] != None:
                    stripped_domains[dix] = domain[:].strip('!('+new_masks[dix]+')')
            except ValueError as e:
                print('WARNING: Trajectory domain {} could not be ' 
                        'stripped further. This usually happens when ' 
                        'the trajectory contains fewer residues than ' 
                        'the reference structure!'.format('A' if dix == 0 else 'B'))         

        
        ref_coords_A, ref_coords_B = calc.standard_orientation(
                stripped_refs, new_mask_A, new_mask_B, args.output)
    else:
        ref_coords_A, ref_coords_B = calc.standard_orientation(
                stripped_domains, None, None, args.output)


    ### Extract data for the two domains
    # TrajectoryIterator.strip as used here returns the trajectory stripped
    # down to the masks given in args.A and args.B. 
    # This is somehow faster then applying masks to a normal trajectory object.
    
    print('Preparing structural data from input.')
    
    traj_A = stripped_domains[0]
    traj_B = stripped_domains[1]
    coords_A = traj_A.xyz
    coords_B = traj_B.xyz
    results = []
    vectors_all, distances_all =[],[] #These are for the vmd visualization
    
    ### Calculate angles for each frame      
    print('Calculating angles.')

    
    # Enumerate over all frames: Not very efficient, 
    # but the time limiting step is anyway the initiation of the trajectory.
    for i, frame in enumerate(coords_A):
        
        B_points, A_points = calc.apply_coordinatesystem(ref_coords_A,
                                                         ref_coords_B,
                                                         coords_A[i],
                                                         coords_B[i],
                                                         ref_com_A,
                                                         ref_com_B) 
        
        # Calculate Angles, add time and RMSD to output
        angles = list(calc.angle_calculation(B_points, A_points))
        rmsd = list(calc.orientational_rmsd(ref_coords_A,
                                            ref_coords_B,
                                            coords_A[i],
                                            coords_B[i]))
        simulated_time = [float(args.simtime)/len(coords_A)*i]
        
        results.append(angles + rmsd + simulated_time)
        
        ### Calculate some data needed for the vmd visualization from each frame
        if args.vmd or args.pymol: 
            vectors_frame = []
            distance_frame = []
            for domain in (B_points, A_points):
                start_point = domain[0]
                distance_frame.append(list(start_point))
                for c in range(1,4):
                    try:
                        # start point / end point / vec number
                        vector_data = [list(start_point), list(domain[c]), c] 
                        vectors_frame.append(vector_data)
                    except: 
                        pass
            #append 0 as identifier of the distance axis            
            distance_frame.append(0) 
            vectors_all.append(vectors_frame)
            distances_all.append([distance_frame]) 
            
    ### Output data
    print('Outputing and visualizing data.')
    
    cols = ['AB','AC1','BC1','AC2','BC2','dc','Time', 'RMSD_A','RMSD_B']
    
    df = pd.DataFrame(results, columns=cols)
    df = df.apply(pd.to_numeric, errors='ignore') 
    with open('OCD_{}.dat'.format(args.output), 'w+') as f:
        df.to_csv(f, sep='\t',index=False, float_format='%.3f')
          
    # Write a TCL script for the visualization in VMD - 
    # needs an aligned trajectory to work, so it outputs one as well    
    if args.vmd:
        vis.vmd_script('./OCD_{}_vmd_ocd.nc'.format(trajname), 
                        './OCD_{}_vmd_ocd.parm7'.format(trajname), 
                        (0,-1,1), 
                        vectors_all, 
                        distances_all, 
                        args.output)
        
        if not os.path.isfile('./OCD_{}_vmd.nc'.format(trajname)):
                print('Writing trajectory for VMD visualization...')
                pt.write_traj('./OCD_{}_vmd_ocd.nc'.format(trajname),
                              traj_all,
                              overwrite=True)
                pt.write_parm('./OCD_{}_vmd_ocd.parm7'.format(trajname),
                              top=traj_all.top,
                              overwrite=True)
    
    # Write out first frame as pdb for PyMol visualization
    if args.pdb or args.pymol: 
        frame1 = pt.iterframe(traj_all, frame_indices=[0])
        frame1.save('./OCD_{}.pdb'.format(args.output), overwrite=True)
        
        # Write PyMOl input script for vizaulisation of 
        # the first frame and coordinate system
        if args.pymol: 
            str_o = ''
            str_o += vis.pymol_init()
            str_o += vis.pymol_load('./OCD_{}.pdb'.format(args.output), 
                                    'Frame1')
            str_o += vis.pymol_draw_vectors(vectors_all[0], 
                                            distances_all[0])
            str_o += vis.pymol_settings()

            with open('./OCD_{}.pym'.format(args.output), 'w+') as f:
                f.write(str_o)

    if args.plot:
        labeldict = {'AB':'AB Angle /$^\circ$',
                     'AC1':'AC1 Angle /$^\circ$',
                     'BC1':'BC1 Angle /$^\circ$',
                     'AC2':'AC2 Angle /$^\circ$',
                     'BC2':'BC2 Angle /$^\circ$',
                     'dc':'dc Distance /$\AA$'}
        titledict = {'AB':'AB Torsion Angle',
                     'AC1':'AC1 Tilt Angle',
                     'BC1':'BC1 Tilt Angle',
                     'AC2':'AC2 Tilt Angle',
                     'BC2':'BC2 Tilt Angle',
                     'dc':'dc Distance'}
        lim_dict = {'AB':args.lim_AB,
                    'AC1':args.lim_AC1,
                    'BC1':args.lim_BC1,
                    'AC2':args.lim_AC2,
                    'BC2':args.lim_BC2,
                    'dc':args.lim_DC}
        
        for s in ['AB','AC1','BC1','AC2','BC2','dc']:
            if s == 'dc':
                binw = 0.1
            else:
                binw = 0.5
                
            vis.angle_plots(x_data=df[s], 
                            y_time= df['Time'], 
                            xlim= lim_dict[s],
                            xlabel = labeldict[s],
                            title = titledict[s],
                            bin_dims=(np.floor(df[s].min()),
                                      np.ceil(df[s].max()), binw),
                            norm=True, 
                            hist2c = '#08519c',
                            linec = 'darkblue', 
                            plot_type = ['hist', 'time'])
            
            if not os.path.exists('./OCD_{}_Plots'.format(args.output)):
                os.makedirs('./OCD_{}_Plots'.format(args.output))
            plt.savefig('./OCD_{}_Plots/{}.png'.format(args.output, s), 
                        bbox_inches='tight')
        
    calc.sysexit(0)
    
### Run that script! 
if __name__ == "__main__":
    main()       
