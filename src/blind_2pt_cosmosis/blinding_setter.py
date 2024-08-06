import datetime
import argparse
from pathlib import Path
from .io import save_pickle, load_pickle, generate_key, load_key, DictAction, DEFAULT_PARAM_RANGE
from .param_shifts import draw_flat_param_shift_mult

def get_parser():
    """
    creates the parser to obtain the user command line input
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=f'''
    --------------------------------------------------------------------------------
    Blinding Module for 2pt data in Cosmosis.
    
    --------------------------------------------------------------------------------
    This module will apply a blinding factor to 2pt functios stored in a fits
    file, using cosmosis to compute the blinding factors.

    The workhorse function that puts everything together is do2ptblinding().
    This script can be called with command line arguments, or by calling that
    function in another script.

    The script uses cosmosis, so you'll need to  source its setup file before
    running this.

    ............................................................
    What it does:
    1. Using a string seed, pseudo-randomly draw a shift in parameters. This
    will eitherbe drawn from a flat distribution in some predefined parameter
    ranges, or from  a distribution defined in paramshiftmodule. Return
    dictionary of shifts where keys are parameter names matching those expected
    by run_cosmosis_togen_2ptdict.
    -> See draw_paramshift

    2. Using a cosmosis parameter ini file template, compute 2pt functions at
    reference and shifted cosmologies by running the cosmosis  twice. This
    should work no matter what sampler shows up in the template ini file.
    (If it uses the test sampler,  make sure the output is nothing, so that
    it doesn't save the cosmology and 2pt info.) Gets the 2pt functions into
    dictionaries of a specific format.
    -> See run_cosmosis_togen_2ptdict

    4. Take ratio or difference of 2pt datavectors to get blinding factor, in same
    dictionary format. (NOTE, MULTIPLICATIVE BLINDING IS CURRENTLY DISABLED)
    -> See get_factordict

    5. Apply blinding factor to desired datafile, saving a new fits file with
    string key in filename.
    -> See apply2ptblinding_tofits
    ............................................................
    ''', fromfile_prefix_chars='@')


    parser.add_argument('-s', '--seed', type=str, required=False,
                        default="HARD_CODED_BLINDING",
                        help='string used to seed parameter shift selection')

    parser.add_argument('-n', '--nblinds', type=int, required=False,
                        default=3,
                        help="Number of blinds to use, one of which is the truth. Default is 3")
    
    parser.add_argument('-p', '--paramshifts', action=DictAction, required=False,
                        default=DEFAULT_PARAM_RANGE,
                        help="Dictionary of parameter shifts between quotes \". \nPlease use the parameter names" +
                        "as named in Cosmosis. \n>> Default is \"{'cosmological_parameters--sigma8_input':(0.834-3*.04,0.834"+
                        "+3*0.04),\ncosmological_parameters--w':(-1.5,-.5)}\"")
    return parser

if __name__ == '__main__':
    # Usage: python blinding_setter.py parameter -p paramshifts -n nblinds [-s seed]
    #   e.g. python blinding_setter.py -p "{'cosmological_parameters--sigma8_input':(0.834-3*.04,0.834
    #                                      +3*0.04), 'cosmological_parameters--w':(-1.5,-.5)}\" -n 3 -s 130963
    
    
    # Read the maximum value of |k| and the random number seed from command line
    # Read in user input
    
    print('Important Notice to External Blinder:')
    print('Retain this screen output from this code for unblinding.')
        
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    nblinds = args.nblinds
    paramshifts = args.paramshifts
    
    print(f'Parameters and ranges in which parameter shifts will be drawn = {paramshifts}')
    print(f'Random seed = {seed}')
    
    
    params_shifts = draw_flat_param_shift_mult(seed, paramshifts, nblinds)
    
    for key in params_shifts.keys():
        print(f'{key} = {params_shifts[key]}')
    
    # Encrypt
    
    # First we generate a key and save it to a file
    
    today = datetime.date.today()
    
    generate_key(f'{Path.home()}/blinding_key_{today}.txt')
    save_pickle, load_pickle, generate_key, load_key
    
    print('Encoding shifts to a pickle for blinding script.')
    
    save_pickle(params_shifts, f'{Path.home()}/blinded_params_{today}.pkl', key_path=f'{Path.home()}/blinding_key_{today}.txt')
    
    # Check if decoding works
    decoded = load_pickle(f'{Path.home()}/blinded_params_{today}.pkl', key_path=f'{Path.home()}/blinding_key_{today}.txt')
    
    print('Decoded dictionaries == original?')

    for key in decoded.keys():
        print(f'{key} = {params_shifts[key]}')
    
    
    

    


