import numpy as np
import os
import logging
from astropy.io import fits
import shutil
from scipy.interpolate import interp1d
from cosmosis.datablock import BlockError
import sacc

logger = logging.getLogger("2pt_blinding")

default_sections = {
    # Spectrum default sections
    "cl_ee": ("spectrum", "shear_cl",),
    "galaxy_shear_cl_ee": ("spectrum", "shear_cl",),
    "cl_bb": ("spectrum", "shear_cl_bb",),
    "galaxy_shear_cl_bb": ("spectrum", "shear_cl_bb",),
    "cl_eb": ("spectrum", "shear_cl_eb",),
    "galaxy_shear_cl_eb": ("spectrum", "shear_cl_eb",),
    "cl_be": ("spectrum", "shear_cl_be",),
    "galaxy_shear_cl_be": ("spectrum", "shear_cl_be",),
    "galaxy_density_cl": ("spectrum", "galaxy_cl",),
    "galaxy_shearDensity_cl_e": ("spectrum", "galaxy_shear_cl",),
    "galaxy_shearDensity_cl_b": ("spectrum", "galaxy_shear_cl",),

    # Real default sections
    "xi_ee": ("real", "shear_xi",),
    "galaxy_shear_xi_ee": ("real", "shear_xi",),
    "xi_bb": ("real", "shear_xi_bb",),
    "galaxy_shear_xi_bb": ("real", "shear_xi_bb",),
    "xi_eb": ("real", "shear_xi_eb",),
    "galaxy_shear_xi_eb": ("real", "shear_xi_eb",),
    "xi_be": ("real", "shear_xi_be",),
    "galaxy_shear_xi_be": ("real", "shear_xi_be",),
    "galaxy_shear_xi_minus": ("real", "shear_xi_minus",),
    "galaxy_shear_xi_plus": ("real", "shear_xi_plus",),
    "galaxy_shear_xi_imagMinus": ("real", "shear_xi_minus",),
    "galaxy_shear_xi_imagPlus": ("real", "shear_xi_plus",),
    "galaxy_density_xi": ("real", "galaxy_xi",),
    "galaxy_shearDensity_xi_t": ("real", "galaxy_shear_xi",),
    "galaxy_shearDensity_xi_x": ("real", "galaxy_shear_xi",),

    # COSEBI's and Psi-stats default sections
    "galaxy_shear_cosebi_bb": ("cosebis", "cosebis_b"),
    "galaxy_shear_cosebi_ee": ("cosebis", "cosebis"),
    "galaxy_shearDensity_cosebi_e": ("cosebis", "psi_stats_gm"),
    "galaxy_density_cosebi": ("cosebis", "psi_stats_gg"),

    # One-point default sections
    # Come back and think about the naming here later:
    "galaxy_stellarmassfunction": ("one_point_mass", "smf",),
    "galaxy_luminosityfunction": ("one_point_luminosity", "lf",),
}

class SpectrumInterp(object):
    """
    This is copied from 2pt_like, for get_data_from_dict_for_2pttype

    Should not get used if bin averaging is being used; is in place
    for if theory vector in datablock is very densely sampled, this 
    gets used to pick out values corresponding to desired angle positions
    """
    def __init__(self,angle,spec,bounds_error=False):
        if np.all(spec>0):
            self.interp_func=interp1d(np.log(angle),np.log(spec),bounds_error=bounds_error,fill_value=-np.inf)
            self.interp_type='loglog'
        elif np.all(spec<0):
            self.interp_func=interp1d(np.log(angle),np.log(-spec),bounds_error=bounds_error,fill_value=-np.inf)
            self.interp_type='minus_loglog'
        else:
            self.interp_func=interp1d(np.log(angle),spec,bounds_error=bounds_error,fill_value=0.)
            self.interp_type="log_ang"

    def __call__(self,angle):
        if self.interp_type=='loglog':
            spec=np.exp(self.interp_func(np.log(angle)))
        elif self.interp_type=='minus_loglog':
            spec=-np.exp(self.interp_func(np.log(angle)))
        else:
            assert self.interp_type=="log_ang"
            spec=self.interp_func(np.log(angle))
        return spec
    
def extract_one_point_prediction(sacc_data, block, data_type, section, **kwargs):
    category = kwargs.get("category")
    tracer_tuples = sacc_data.get_tracer_combinations(data_type)

    theory_vector = []
    observable_vector = []

    for t in tracer_tuples:
        assert len(t) == 1, "One-point likelihoods only support single tracer data types"
        # This should be an n(z) tracer
        tracer = t[0]
        b = int(tracer.split("_")[1]) + 1
 
        # Determine the key for accessing block and sacc_data based on category
        # This selection below needs to be generalised more 1pt statistics we implement (cluster richness for instance, ...).
        key = "mass" if category == "one_point_mass" else "luminosity"
        x_theory = block[section, f"{key}_{b}"]
        theory = block[section, f"bin_{b}"]
        theory_spline = interp1d(x_theory, theory, bounds_error=False, fill_value="extrapolate")

        window = None
        for w in sacc_data.get_tag("windows", data_type, t):
            if (window is not None) and (w is not window):
                raise ValueError("Sacc likelihood currently assumes data types share a window object")
            window = w

        # TO-DO: Check if the window thing is ok for 1pt stats and how to do the binning here either way.
        x_nominal = np.array(sacc_data.get_tag(key, data_type, t))
        
        if window is None:
            binned_theory = theory_spline(x_nominal)
        else:
            x_window = window.values
            theory_interpolated = theory_spline(x_window)
            index = sacc_data.get_tag("window_ind", data_type, t)
            weight = window.weight[:, index]

            # The weight away should hopefully sum to 1 anyway but we should
            # probably not rely on that always being true.
            # TO-DO: Check this for real statistics, but should be ok.
            binned_theory = (weight @ theory_interpolated) / weight.sum()
    
        theory_vector.append(binned_theory)
        observable_vector.append(x_nominal)
        
    theory_vector = np.concatenate(theory_vector)
    observable_vector = np.concatenate(observable_vector)

    return theory_vector, observable_vector

def extract_spectrum_prediction(sacc_data, block, data_type, section, **kwargs):

    category = kwargs.get("category")
    if category == "spectrum":
        x_theory = block[section, "ell"]
    elif category == "real":
        x_theory = block[section, "theta"]
    #TO-DO: Decide on final nomenclature for cosebis and psi-stats!
    # Given current cosebis module in standard library, the x_nominal should be simply n
    elif category == "cosebis":
        x_theory = block[section, "n"]
    is_auto = block[section, "is_auto"]

    # We build up these vectors from all the data points.
    # Only the theory vector is needed for the likelihood - the others
    # are for convenience, debugging, etc.
    theory_vector = []
    angle_vector = []

    # Because we called to_canonical_order when we loaded the data,
    # we know that the data is grouped by data type, and then by tracers (tomo bins).
    # So that means we can do a data type at a time and then concatenate them, and
    # within this do a bin pair at a time, and concatenate them too.
    for b1, b2 in sacc_data.get_tracer_combinations(data_type):
        # Here we assume that the bin names are formatted such that
        # they always end with _1, _2, etc. That isn't always true in
        # sacc, but is somewhat baked into cosmosis in other modules.
        # It would be nice to update that throughout, but that will
        # have to wait. Also, cosmosis bins start from 1 not 0.
        # We need to make sure that's fixed in the 
        i = int(b1.split("_")[-1]) + 1
        j = int(b2.split("_")[-1]) + 1

        if 'shearDensity' in data_type:
            i, j = j, i

        try:
            theory = block[section, f"bin_{i}_{j}"]
        except BlockError:
            if is_auto:
                theory = block[section, f"bin_{j}_{i}"]
            else:
                raise

        # check that all the data points share the same window
        # object (window objects contain weights for a set of ell / theta values,
        # as a matrix), or that none have windows.
        window = None
        for d in sacc_data.get_data_points(data_type, (b1, b2)):
            w = d.get_tag('window')
            if (window is not None) and (w is not window):
                raise ValueError("Sacc likelihood currently assumes data types share a window object")
            window = w

        # We need to interpolate between the sample ell / theta values
        # onto all the ell / theta values required by the weight function
        # This will give zero outside the range where we have
        # calculated the theory
        theory_spline = SpectrumInterp(x_theory, theory)
        if window is not None:
            x_window = window.values
            theory_interpolated = theory_spline(x_window)

        for d in sacc_data.get_data_points(data_type, (b1, b2)):
            if category == "spectrum":
                x_nominal = d['ell']
            elif category == "real":
                x_nominal = d['theta']
            #TO-DO: Decide on final nomenclature for cosebis and psi-stats!
            # Given current cosebis module in standard library, the x_nominal should be simply n
            elif category == "cosebis":
                x_nominal = d['n']

            if window is None:
                binned_theory = theory_spline(x_nominal)
            else:
                index = d['window_ind']
                weight = window.weight[:, index]

                # The weight away should hopefully sum to 1 anyway but we should
                # probably not rely on that always being true.
                # TO-DO: Check this for real statistics, but should be ok.
                binned_theory = (weight @ theory_interpolated) / weight.sum()

            theory_vector.append(binned_theory)
            angle_vector.append(x_nominal)

    theory_vector = np.array(theory_vector)
    angle_vector = np.array(angle_vector)

    return theory_vector, angle_vector

def get_twoptdict_from_pipeline_data_sacc(block, sacc_file):
    """
    Extract 2pt data from a cosmosis pipeline data object and return a dictionary
    with keys corresponding to the 2pt spectra types.
    """
    outdict = {}
    theory = []
    angle = []
    sacc_data = sacc.Sacc.load_fits(sacc_file)
    # Now we actually loop through our data sets
    for data_type in sacc_data.get_data_types():
        category, section = default_sections[data_type]
        if "one_point" in category:
            theory_vector, angles = extract_one_point_prediction(sacc_data, block, data_type, section, category=category)
        else:
            theory_vector, angles = extract_spectrum_prediction(sacc_data, block, data_type, section, category=category)

        theory.append(theory_vector)
        angle.append(angles)
    
    return np.concatenate(theory)
    
def apply_2pt_blinding_and_save_sacc(factordict, origsaccfile, mode='xi', outfname=None, outftag="_BLINDED",
                                     justfname=False, bftype='add', storeseed='notsaved'):
    """
    Given the dictionary of one set of blinding factors,
    the name of a  of a fits file containing unblinded 2pt data,
    and (optional) desired output file name or tag,
    multiplies 2pt data in original fits file by blinding factors and
    saves results (blinded data) into a new fits file.

    If argument is passed for outfname, that will be the name of output file,
    if not, will be <input filename>_<outftag>.fits. The script will check
    that the outfname is different than the original, unblinded file. If they
    match, it will revert to default behavior for naming the blinded file
    so that the unblinded file isn't overwritten.
    > if storeseed=True, will save whatever string is there to entry in fits file

    If justfname == True, doesn't do any file manipulation, just returns
    string of output filename. (for testing)

    bftype can be 'add','mult' or 'mult-nocs'. For data vec d, blinding factor f
        'add' - do additive blinding:
                    d_blind = d_input + f, f = d_shift - d_ref, cov_bl = cov
        'mult' - do multiplicative blinding, scale covmat in blinded file:
                    d_blind = d_input*f, f = d_shift/d_ref, cov_bl = f^T*cov*f
        'multNOCS' - do multiplicative blinding, but without covariance scaling
                    d_blind = d_input*f, f = d_shift/d_ref, cov_bl = cov
    """

    sacc_data_orig = sacc.Sacc.load_fits(origsaccfile)
    logger.info(f'apply2ptblinding for: {origsaccfile}')
    # check whether data is already blinded and whether Nbins match
    if 'blinded' in sacc_data_orig.metadata: 
        # check for blinding
        # if entry not there, or storing False -> not already blinded
        raise ValueError('Data is already blinded!')

    # set up output file
    if outfname == None or outfname==origsaccfile:
        #make sure you can't accidentaly overwrite the original
        #if output filename isn't given, add outftag onto the name of the
        #  unblinded file
        if outftag == None or outftag =='':
            outftag = '_BLINDED-defaulttag'
        outfname = origsaccfile.replace('.sacc','{0}.sacc'.format(outftag))

    if not justfname:
        sacc_data_blinded = sacc.Sacc.load_fits(origsaccfile)

         #apply blinding factors
        if bftype=='mult' or bftype=='multNOCS':
            sacc_data_blinded.mean = sacc_data_orig.get_mean() * factordict
            if bftype=='mult': 
                raise ValueError('Not currently set up to do covariance scaling needed for multiplicative blinding.')
        elif bftype=='add':
            sacc_data_blinded.mean = sacc_data_orig.get_mean() + factordict
        else:
            raise ValueError('bftype {0:s} not recognized'.format(bftype))

        sacc_data_blinded.metadata = sacc_data_orig.metadata
        # adds metadata to the sacc file:
        sacc_data_blinded.metadata['blinded'] = True
        sacc_data_blinded.metadata['info'] = 'Blinded data-vector.'
        sacc_data_blinded.metadata['keyword'] = storeseed
        sacc_data_blinded.save_fits(outfname, overwrite=True)
            
        logger.info(f">>>> Stored blinded data in {outfname}")
    return outfname

