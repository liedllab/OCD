#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vhoer

Functions used by OCD.py for calculations.
"""

import pytraj as pt
import numpy as np
import sys
import os
import datetime
import pandas as pd
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from mdtraj.geometry import alignment as al
from mdtraj.geometry.alignment import compute_transformation

###################
### Vector math ###
###################

def unit_vec(vec):
    '''
    Calculates unit vector of any vector provided
    
    Parameters:
    -----
    Requires:
        vec; a numpy array of shape (n), which represents a vector of arbitrary length
    
    Returns:
        unit_vec, a numpy array of shape (n), which represents a vector of length 1
    '''
    return np.array(vec/np.linalg.norm(vec))

def dist(a, b):
    """
    Distance between two vectors.
    
    Parameters:
    -----
    Required:
        a, b; numpy arrays of shape (n) containing an vector with n entries
    
    Returns:
        dist, a float giving the euclidean distance between vectors a and b
    
    """
    return np.linalg.norm(b-a)

def dihedral_vectors(b0, b1, b2):
    '''
    Function to calculate dihedral angles from three vectors 1 -> 2 -> 3
    
    Praxeolitic formula
    1 sqrt, 1 cross product
    
    From https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    Fastest of all dihedral functions in the above stackexchange answer
    
    Parameters:
    -----
    Requires:
        b0, b1, b2; numpy arrays of shape (3), containing three vectors pointing from point 2 to 1 (b0), 2 to 3 (b1) and 3 to 4 (b2)
        
    Returns:
        the torsion angle around b1 in degrees
    '''

    ### normalize b1 so that it does not influence magnitude of vector rejections that come next
    b1 /= np.linalg.norm(b1)

    ### vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    ### angle between v and w in a plane is the torsion angle, v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def full_angle(v1,v2,v3):
    '''
    Calcualtes angle between two vectors v1 and v2. 
    Increases range of angle from [0°,180°] to [-180°,180°] by comparing orientation with vector v3. 
    v3 should be not parallel to v1/v2 plane 
    Requires:
        v1, v2; numpy arrays of shape (3), containing three vectors for which angle is calculated
        v3; numpy array of shape (3), containing vector approx. normal to plane formed by v1 and v2
    
    Returns:
        a, angle in degrees as float
    '''
    a = np.arccos( np.dot(v1, v2) ) 
    cross = np.cross(v1, v2)
    if np.dot(v3, cross) < 0:
        a = -a
    return np.degrees(a)

#################################################################################
### Functions pertaining to the moment of inertia tensor and it's eigenvectors ###
#################################################################################

def compute_inertia_tensor(coords, masses):
    """
    Function appropriated from mdtraj.geometry.order.
    Compute the inertia tensor of a trajectory.

    For each frame,
        I_{ab} = sum_{i_atoms} [m_i * (r_i^2 * d_{ab} - r_{ia} * r_{ib})]
    ...but doing this as an Einstein summation is faster!
    
    Parameters
    ----------
    traj : Trajectory
        Trajectory to compute inertia tensor of.

    Returns
    -------
    I_ab:  np.ndarray, shape=(traj.n_frames, 3, 3), dtype=float64
        Inertia tensors for each frame.
    """

    eyes = np.empty(shape=(len(coords), 3, 3), dtype=np.float64)
    eyes[:] = np.eye(3)
    A = np.einsum("i, kij->k", masses, coords ** 2).reshape(len(coords), 1, 1)
    B = np.einsum("ij..., ...jk->...ki", masses[:, np.newaxis] * coords.T, coords)
    return A * eyes - B

def choose_evec(evec, com, point):
    '''
    Get rid of eigenvector sign ambiguity by checking distance to a fixed point. 
    Will return either evec or -evec, whichever added to com (com+evec or com-evec) is closer to point
    
    Parameters:
    -----
    Required:
        evec; a numpy array of shape (3) containing a vector with ambigous sign
        com; a numpy array of shape (3) representing the xyz coordinates of the origin for the vector evec
        point; a numpy array of shape (3) giving the xyz coordinates of a point towards which the vector will be oriented
        
    Returns:
        evec_oriented; a numpy array of shape (3) containing the vector which sign has been choosen such that it points towards point
    '''

    if dist(com+evec, point) < dist(com-evec,point):
        return evec
    else:
        return np.negative(evec)
    
def orient_evecs(evecs, com, point_lowest, point_middle):
    '''
    Force evec orientation into a right-handed coordinate system by orientating the second and third evec according to the distance to a choosen point and the first one to be in the same diretion as the dot product of the others.
    
    Parameters:
    -----
    Requires:
        evecs; a numpy array of shape (3,3) containing the eigenvectors of a moment of inertia tensor. Eigenvectors should be provided as row vectors and be ordered from highest to lowest eigenvalue.
        point_lowest; a numpy array of shape (3) containing xyz coordinates of a point towards which the lowest eigenvector will be orientated.
        point_middle; a numpy array of shape (3) containing xyz coordinates of a point towards which the middle eigenvector will be orientated.
    
    Returns:
        corrected_evecs; a numpy array of shape(3,3) containing the reoriented eigenvectors as row vectors
    '''
    corrected_evecs = np.zeros_like(evecs)
    corrected_evecs[2] = choose_evec(evecs[2], com, point_lowest)
    corrected_evecs[1] = choose_evec(evecs[1], com, point_middle)
    if np.array_equal(np.sign(np.cross(corrected_evecs[1], corrected_evecs[2])), np.sign(evecs[0])):
        corrected_evecs[0] = evecs[0]
    else: 
        corrected_evecs[0] = np.negative(evecs[0])
    return corrected_evecs   

#################################################################################
### Functions to reorient the reference structure into a standard orientation ###
#################################################################################

def new_masks(stripped_domains, stripped_ref_domains):
    '''
    This function compares the residues found in the sample and the reference structure after the atom mask was applied.
    If reference structure and traj have a different amount of atoms, we assume this to be due to missing residues.
    Therefore, we generate a new mask corresponding to only the residues shared between them. This goes off the residue names in the pdb files.
    Note that this won't work correctly with trajectories, as the residues there are just renumbered to start from 1 in all cases.
    
    Parameters:
    -----
    Requires:
        stripped_domains; a list containing two pytraj trajectory objects - each of them containing a domain of the sample structure 
        stripped_ref_domains; a list containing two pytraj trajectory objects - each of them containing a domain of the reference structure 
        
    Returns:
        new_mask_a; a mask string in pytraj notation containing all the shared residues between sample and reference in domain A
        new_mask_b; a mask string in pytraj notation containing all the shared residues between sample and reference in domain B
    '''
    final_residues = []
    if stripped_domains[0].xyz.shape[1] != stripped_ref_domains[0].xyz.shape[1] or stripped_domains[1].xyz.shape[1] != stripped_ref_domains[1].xyz.shape[1]:
        print('WARNING: Found different amount of atoms in the trajectory and the reference. Cutting down residues...')
        for tix, traj in enumerate(stripped_domains):
            ref = stripped_ref_domains[tix]
            ref_res = set([res.original_resid for res in ref.top.residues])
            traj_res = set([res.original_resid for res in traj.top.residues])          
            if traj_res == ref_res:
                final_residues.append(None)
            else:
                common_residues = traj_res.intersection(ref_res)
                final_residues.append(':;'+','.join([str(i) for i in common_residues]))


        print('New Mask A:\t'+str(final_residues[0]))
        print('New Mask B:\t'+str(final_residues[1]))        
        
    else: #if they have the same amount of atoms anyway, just do nothing (these None masks are okay to use because we strip from domains everything but these masks, not select anything)
        final_residues=(None, None)
    
    return final_residues[0], final_residues[1]    
     
def standard_orientation(stripped_domains, mask_A, mask_B, out):
    '''
    Calculates standard orientation for two domains given as masks A and B applied to trajectory traj. 
    Standard orientations are calculated as the orientation where the lowest moment of inertia eigenvector of the domain is aligned with z, the x axis is as close to the vector bewtween the centers of mass of the two domains as possible.
    The y vector is calculated as the negative crossproduct of the z-vector (lowest eigenvector) and x-vector(projection of com vector), giving a right handed coordinate system.
    The whole domain is alligned to this coordinate system in such a way that the center of mass is in the origin.
    The coordinates of the two alligned domains are returned.
    
    The full reference structure is used for the calculation of the coordinate system, but the coordinates are stripped before output according to masks A/B -> coordinate system only dependant on refstructure, not on the necessary mask for further alignment
    This is necessary, because crystal structure files might contain fewer residues than the reference structure which would mess up the alignment
    
    Parameters
    ----
    Requires:
        stripped_domains, a list containing two pytraj trajectories corresponding to domains A (entry 0) and B (entry 1)
        mask_A, a str giving the cpptraj/pytraj mask corresponding to domain A in the trajectory - note that this mask 
        mask_B, a str giving the cpptraj/pytraj mask corresponding to domain B in the trajectory
    
    Returns:
        transformed_coordinates_A; a numpy array of shape (n_atoms, 3), giving the reoriented coordinates of domain A to be used as reference orientation for ABangle calculations
        transformed_coordinates_B; a numpy array of shape (n_atoms, 3), giving the reoriented coordinates of domain B to be used as reference orientation for ABangle calculations
        
    '''
    ### Apply both masks and calculate masses, com, coordinates and a reference point
   
    coms = [pt.center_of_mass(stripped_domains[0]), pt.center_of_mass(stripped_domains[1])]
    x_vec = [1, 0, 0]
    y_vec = [0, 1, 0]
    z_vec = [0, 0, 1]
    origin = [0, 0, 0]
    masks = [mask_A, mask_B]
    unit_vectors = np.array([x_vec, y_vec, z_vec, origin]) #Orderd xyz
    results = []
    
    for i, domain in enumerate(stripped_domains):    
        
        partner = 0 if i==1 else 1 #selects index of partnered domain
        mask = masks[i]

        ### Strip down coordinates to mask, does not affect calculation of coordinate system / moment of inertia tensor, just the reference coordinate data to be returned
        ### This is used in case the reference structure has more residues in it than the sample structure (see function new_maks)
        try:
            if mask != None:
                traj_stripped = domain[:].strip('!('+mask+')' )
            else:
                traj_stripped = domain
        except: 
            print('WARNING: Reference structure could not be further stripped. This generally happens if the trajectory contains more residues than the reference structure. ')
            traj_stripped = domain
        
        traj_partner = stripped_domains[partner]
        
        masses = domain.top.mass
        com_traj = coms[i]
        com_partner = coms[partner]

        ref_coordinates = domain.xyz

        ref_point = domain.xyz[0,0]

        inertia_tensor = compute_inertia_tensor(ref_coordinates- np.expand_dims(com_traj,axis=1), masses)
        evals, evecs = np.linalg.eigh(inertia_tensor[0])
        
        ### Reorder evecs to conform to xyz orientation. Largest evec should be x, lowest z
        evals = evals[::-1]
        evecs = evecs.T[::-1]
        evecs = orient_evecs(evecs, com_traj[0], ref_point, com_partner[0])
        evec_orientation = np.append(evecs+com_traj[0],[com_traj[0]], axis=0) #add center of mass to evec array to conform to xyz+origin ordering, same as unit_vectors
       
        ### Apply first transformation: Aligns eigenvectors of inertia tensor with xyz unit vectors. The center of mass is now in the origin
        transformation = compute_transformation(evec_orientation, unit_vectors)
        z_aligned_ref_coords = transformation.transform(traj_stripped.xyz[0])
         
        ### Calculate vectors for second transformation: new_com_partner is the transformed partner com, which is then projected onto the xy plane and stored as a unitvector.
        new_com_vector = transformation.transform(com_partner[0])
        new_com_vector[2] = 0 #projection on xy plane
        new_com_vector = unit_vec(new_com_vector)
        orthogonal_vector = -np.cross(new_com_vector, z_vec)  
        
        ### Check angle of z-vec to com vector to make sure the coordinate system makes sense, they should be roughly orthogonal 
        if abs(np.dot(z_vec, new_com_vector)) > 0.25:
            print('WARNING: Ange between center axis and Vectors in domain {} is lower than $\pm$ 75 degrees. Please consider checking visualizations of the coordinate system to ensure that this works for your purpose.')
        
        ### Do second transformation. Lowest eigenvector is still z, x is now the projection of the com vector -> closest possible alignment of x and com while keeping x orthogonal to z
        reference_orientation = np.array([unit_vec(new_com_vector), unit_vec(orthogonal_vector), z_vec, origin])
        second_transformation = compute_transformation(reference_orientation, unit_vectors)
        x_z_aligned_ref_coords = second_transformation.transform(z_aligned_ref_coords)
        results.append(x_z_aligned_ref_coords)
        
    transformed_coordinates_A = results[0]
    transformed_coordinates_B = results[1]   

    return transformed_coordinates_A, transformed_coordinates_B #Reference coordinates A and B
        
###############################################
### Functions to calculate the OCD measures ###
###############################################


def angle_calculation(B_points, A_points):
    '''
    Function to calculate OCD angles from vectors.
    
    Parameters:
    -----
    Requires:
        B_points -> List of 3 numpy arrays of shape (3), containing (in order) the transformed coordinates of the center of mass, the endpoint of the B1 vector and the endpoint of the B2 vector
        A_points -> List of 3 numpy arrays of shape (3), containing (in order) the transformed coordinates of the center of mass, the endpoint of the A1 vector and the endpoint of the A2 vector
    
    Returns:
        OCD_Angles; a list with 6 float entries containing the AB torsion angle, the tilt angles BC1, BC2, AC1, AC2 and the distance dc.
    '''
    
    ### Calculate vectors from given points first
    C = A_points[0] - B_points[0]
    dc = np.linalg.norm(C)
    C = unit_vec(C)
    
    B1 = unit_vec(B_points[1] - B_points[0])
    B2 = unit_vec(B_points[2] - B_points[0])
    
    A1 = unit_vec(A_points[1] - A_points[0])
    A2 = unit_vec(A_points[2] - A_points[0])
    
    ### Calculate angle measures
    AB = dihedral_vectors(B1, C, A1)
    
    BC1 = np.degrees( np.arccos( np.dot(B1, C) ) )
    BC2 = full_angle(C, B2, B1)
    AC1 = np.degrees( np.arccos( np.dot(A1, -C) ) )
    AC2 = full_angle(-C, A2, A1)

    return(AB, AC1, BC1, AC2, BC2,dc)

def apply_coordinatesystem(A_xyz_ref, B_xyz_ref, A_xyz_frame, B_xyz_frame, com_A, com_B, _xvec = np.array([1, 0, 0]), _yvec = np.array([0, 1, 0]), _zvec = np.array([0, 0, 1])):
    """
    Maps the reference frame vectors (coordinate system) onto the A and B domains of an immunoglobulin domain. 
    Assumes that the reference is in the standard orientation (so transformed by standard_orientation): 
        Then the global z-vector corresponds to the chains principal axis of inertia with corresponding lowest eigenvalue, 
        and the x-vector is closest to the COM vector between the two domains while orthogonal to the z-vec.

    Parameters:
    -----
    Required:
        A_xyz_ref -> Numpy array of shape (Nr of atoms, 3), contains xyz coordinates of all A chain atoms of the reference frame/structure
        B_xyz_ref -> Numpy array of shape (Nr of atoms, 3), contains xyz coordinates of all B chain atoms of the reference frame/structure
        A_xyz_frame -> Numpy array of shape (Nr of atoms, 3), contains xyz coordinates of all A chain atoms of the frame/structure from which angles will be calculated to the reference structure
        B_xyz_frame -> umpy array of shape (Nr of atoms, 3), contains xyz coordinates of all A chain atoms of the frame/structure from which angles will be calculated to the reference structure
        com_A -> Numpy array of shape (3), contains xyz coordinates of the A chain center of mass of the !reference structure! (Should be [0,0,0] in the standard orientation)
        com_B -> Numpy array of shape (3), contains xyz coordinates of the B chain center of mass of the !reference structure! (Should be [0,0,0] in the standard orientation)

    Optional:
        _xvec -> Numpy array of shape (1, 3). This vector is used as the third vector of the OCD coordinate system. Use the unit x vector if reference domains are in standard orientation.
        _yvec -> Numpy array of shape (1, 3). This vector is used as the second vector of the OCD coordinate system. Use the unit y vector if reference domains are in standard orientation.
        _zvec -> Numpy array of shape (1, 3). This vector is used as the first vector of the OCD coordinate system. Use the unit z vector if reference domains are in standard orientation.
        
    Returns:
        B_points -> List of 3 numpy arrays of shape (3), containing (in order) the transformed coordinates of the center of mass, the endpoint of the B1 vector and the endpoint of the B2 vector
        A_points -> List of 3 numpy arrays of shape (3), containing (in order) the transformed coordinates of the center of mass, the endpoint of the A1 vector and the endpoint of the A2 vector
    """
    
    # Get transformation matrices by alligning the core of the domains: Reference is aligned onto Frame structure
    A_transform = compute_transformation(A_xyz_ref, A_xyz_frame)
    B_transform = compute_transformation(B_xyz_ref, B_xyz_frame)

    # Add vectors to center of mass to get vector endpoints in the reference domain -> these will then be transformed to the sample domains
    # The vectors are the standard z-vector and y-vector added to the center of mass - see above and in function standard_orientation why!
    
    B1 = com_B + _zvec
    B2 = com_B + _yvec
    B3 = com_B + _xvec
    
    A1 = com_A + _zvec
    A2 = com_A + _yvec
    A3 = com_A + _xvec

    # Apply the transfomation to the reference vectors
    B_points = np.array( B_transform.transform(np.array((com_B,B1,B2,B3))) )
    A_points = np.array( A_transform.transform(np.array((com_A,A1,A2,A3))) )

    return B_points, A_points



#######################
### Other functions ###
#######################

def orientational_rmsd(ref_A, ref_B, coords_A, coords_B):
    '''
    Calculates an RMSD meassure to assess the quality of the alignment of the reference structure. 
    Two RMSD values are calculated, A_rmsd for the alignment of reference A to sample A and B_rmsd for the alignment of reference B to sample B.
    
    Parameters:
    -----
    Requires:
        ref_A; a numpy array of shape (n_atoms, 3), containing the coordinates of reference domain A
        ref_B; a numpy array of shape (n_atoms, 3), containing the coordinates of reference domain B  
        coords_A; a numpy array of shape (n_atoms, 3), containing the coordinates of sample domain A
        coords_B; a numpy array of shape (n_atoms, 3), containing the coordinates of sample domain B 

    Returns:
        A_rmsd; a float describing the RMSD value for the alignment of reference domain A to sample domain A
        B_rmsd; a float describing the RMSD value for the alignment of reference domain B to sample domain B
    '''
    A_rmsd = al.rmsd_kabsch(ref_A, coords_A)
    B_rmsd = al.rmsd_kabsch(ref_B, coords_B)
    return A_rmsd, B_rmsd

def sysexit(i):
    ''' 
    Does the same as sys.exit(i) but prints a different message depending on the current time before exiting.
    '''
    print('')
    if i == 0:
        print('OCD.py finished as expected.')
    elif i != 0:
        print('OCD.py finished with an error. Please try again!')
    time = datetime.datetime.now()
    current_time = str(time.hour)+':'+str(time.minute)
    if time.hour == time.minute:
        print('Oh wow! Right now the time is {!s}. You have great timing and are therefore a great person. Keep up the good work!'.format(current_time))
    if time.hour < 5:
        print('Good night!')
    elif time.hour < 10:
        print('Have a wonderful morning!')
    elif time.hour < 15:
        print('Have a nice day!')
    elif time.hour < 18:
        print('Have a pleasant afternoon!')
    elif time.hour < 22:
        print('Have a great evening!')  
    else:
        print('Have an delightful night!')
    print('\n')
    sys.exit(i)
