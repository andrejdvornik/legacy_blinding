import logging
from .param_shifts import draw_flat_param_shift, get_factordict
from .io import get_parser, get_stored_seed_and_tag, load_pickle
from .run_cosmosis_2pt import run_cosmosis_togen_2ptdict
from .twopt_utils import apply_2pt_blinding_and_save_fits, remove_original_fits_file


def main():
    """
    main function to be called from command line
    """
    # gets the parser from the io module
    parser = get_parser()
    args = parser.parse_args()

    # Configure the logger level based on the input argument
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a logger for the __main__ module
    logger = logging.getLogger("2pt_blinding")
    logger.debug(args)

    # Loop over different blinds, one of which is the truth!
    # Done so by using a query to a encrypted settings with the relevant
    # settings set by another code (blinding_setter.py)
    if args.remotely_blinded:
        encrypted_dict = load_pickle(args.file_path, args.key_path)
        
        for i, blind in enumerate(encrypted_dict.keys()):
            # We read the parameter shifts from externally set encripted pickled file
            #params_shifts = draw_flat_param_shift(seed[i], paramshifts[i])
            #logger.debug(f"Parameters shifts are: {params_shifts}")
            params_shifts = encrypted_dict[blind]
    
            #get blinding factors
            reff_dict = run_cosmosis_togen_2ptdict(inifile=args.ini) #args.ini is a template
            logger.debug(f"Calculated Reference Dict")
            # FIXME: Add nz_file and angles_file to args
            shift_dict = run_cosmosis_togen_2ptdict(inifile=args.ini, pdict=params_shifts,
                                                    nz_file=None, angles_file=None)
            logger.debug(f"Calculated Shifted Dict")
            #gets the blinding factors data vectors
            factor_dict = get_factordict(reff_dict, shift_dict, bftype=args.bftype)
            logger.debug(f"Calculated Blinded Data Vectors")
        
            # Gets some I/O information
            storeseed, tagstr = get_stored_seed_and_tag(args)
        
            # applies the shift to the 2pt data-vector:
            #FIXME: This function is also saving the file, we want to split it!
            apply_2pt_blinding_and_save_fits(factor_dict, origfitsfile=args.origfits,
                                            outfname=args.outfname, outftag=f'_{blind}',
                                            bftype=args.bftype, storeseed=storeseed)
        
        remove_original_fits_file(origfitsfile=args.origfits, remove=args.remove)
                
    else:
        # Will need setting a seed, paramshifts, labels for output catalogues
        params_shifts = draw_flat_param_shift(args.seed, args.paramshifts)
        logger.debug(f"Parameters shifts are: {params_shifts}")
    
        #get blinding factors
        reff_dict = run_cosmosis_togen_2ptdict(inifile=args.ini) #args.ini is a template
        logger.debug(f"Calculated Reference Dict")
        # FIXME: Add nz_file and angles_file to args
        shift_dict = run_cosmosis_togen_2ptdict(inifile=args.ini, pdict=params_shifts,
                                                nz_file=None, angles_file=None)
        logger.debug(f"Calculated Shifted Dict")
        #gets the blinding factors data vectors
        factor_dict = get_factordict(reff_dict, shift_dict, bftype=args.bftype)
        logger.debug(f"Calculated Blinded Data Vectors")
    
        # Gets some I/O information
        storeseed, tagstr = get_stored_seed_and_tag(args)
    
        # applies the shift to the 2pt data-vector:
        #FIXME: This function is also saving the file, we want to split it!
        apply_2pt_blinding_and_save_fits(factor_dict, origfitsfile=args.origfits,
                                        outfname=args.outfname, outftag=tagstr,
                                        bftype=args.bftype, storeseed=storeseed)
                                        
        remove_original_fits_file(origfitsfile=args.origfits, remove=args.remove)

if __name__ == '__main__':
    main()
