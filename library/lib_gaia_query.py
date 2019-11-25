""""
Dedicated Class to query the Gaia archive

Héctor Cánovas May 2019 - now
"""
import warnings, glob
import numpy as np
from astropy.utils.exceptions import AstropyWarning
from astroquery.gaia          import Gaia

# Extra Methods ================================
def read_float_input(text = 'Introduce Input: '):
    while True:
        try:
            read_val = np.float(input(text))
            break
        except ValueError:
            print('This is not a number. Try again...')

    return read_val


def make_col_str(inp_list, first_label = ''):
    out_cols   = [first_label + inp + ',' for inp in inp_list]
    out_cols   = " ".join(out_cols)
    out_cols   = out_cols[:-1]  
    return out_cols


class LibGaiaQuery():
    """
    Initialize the class.
    """
    def __init__(self):
        """
        Define catalogue/column attributes
        """
        # Select Gaia DR2 Cols ===============
        cols_astrom    = ['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'] # Astrometry
        cols_phot_m    = ['phot_' +  inp + '_mean_mag'             for inp in ['g', 'bp', 'rp']]                                # Photometry [mags]
        cols_phot_f    = ['phot_' +  inp + '_mean_flux_over_error' for inp in ['g', 'bp', 'rp']]                                # Photometry [Flux]
        cols_phot_o    = ['l', 'b', 'visibility_periods_used', 'radial_velocity']                                               # Miscellaneus

        # Define WISE Cols ==============
        wise_cols      = ['ph_qual', 'ra as Wra', 'dec as Wdec', 'w1mpro as W1mag', 'w2mpro as W2mag',
                             'w3mpro as W3mag', 'w4mpro as W4mag', 'w1mpro_error as e_W1mag',
                             'w2mpro_error as e_W2mag', 'w3mpro_error as e_W3mag', 'w4mpro_error as e_W4mag']
        # Define 2MASS Cols =============
        tmass_cols     = ['ph_qual', 'j_m as Jmag', 'h_m as Hmag','ks_m as Kmag', 'j_msigcom as e_Jmag', 
                             'h_msigcom as e_Hmag', 'ks_msigcom as e_Kmag']

        self.gaia_cols   = make_col_str(cols_astrom + cols_phot_m + cols_phot_f+ cols_phot_o, first_label = 'gaia.')
        self.gaia_cols_p = make_col_str(['source_id'] + cols_phot_m + cols_phot_f, first_label='gaia.')
        self.wise_cols   = make_col_str(wise_cols,  first_label = 'wise.')
        self.tmass_cols  = make_col_str(tmass_cols, first_label = 'tmass.')


    def __repr__(self):
        """Return developer-friendly string representation."""
        return f'Class to create the Gaia/Analysis sample'


    def set_cone_search_pars(self, sample_name = None, ra = None, dec = None, radii = None, para_m = None, para_M = None):
        """
        Prepare cone search parameters.
        """
        text_0 = 'Preparing Cone-Search ADQL query. Introduce '
        if not sample_name: sample_name = input('Introduce sample name (e.g. IC_348_test): ')
        if not ra:          ra          = read_float_input(text_0 + 'R.A.   [decimal degree]: ')
        if not dec:         dec         = read_float_input(text_0 + 'Dec    [decimal degree]: ')
        if not radii:       radii       = read_float_input(text_0 + 'radius [decimal degree]: ')
        if not para_m:      para_m      = read_float_input(text_0 + 'min-Parallax [mas]: ')
        if not para_M:      para_M      = read_float_input(text_0 + 'max-Parallax [mas]: ')
        print()
        self.sample_name = sample_name
        self.ADQL   = {'ra':ra, 'dec':dec, 'radii':radii, 'para_m':para_m, 'para_M':para_M}


    def run_cone_search(self, quality_par_SN  = '10', quality_par_vis = '7',quality_par_ruwe = '1.40', verbose = True):
        """
        Run a Cone Search ADQL query on the Gaia DR2 archive. By default the code applies a selection criteria: objets with
        parallax S/N <10, visibility periods used < 7, and RUWE >1.40 are excluded.
        """
        # 1.- Define ADQL query ================================
        query  = ("SELECT " + self.gaia_cols + " "
                ",sqrt(gaia.astrometric_chi2_al/(gaia.astrometric_n_good_obs_al-5)) as unit_weight_e, g_ruwe.ruwe as ruwe "
                "FROM gaiadr2.gaia_source as gaia "
                "LEFT OUTER JOIN gaiadr2.ruwe  AS g_ruwe ON gaia.source_id = g_ruwe.source_id "
                "WHERE 1=CONTAINS( "
                "POINT('ICRS',ra,dec), "
                f"CIRCLE('ICRS',{self.ADQL['ra']:5.2F}, {self.ADQL['dec']:5.2F}, {self.ADQL['radii']:5.2F})) "
                f"AND parallax >= {self.ADQL['para_m']:5.2F} AND parallax <= {self.ADQL['para_M']:5.2F} "
                f"AND gaia.source_id IS NOT NULL AND gaia.parallax/gaia.parallax_error >{quality_par_SN} "
                f"AND gaia.visibility_periods_used >{quality_par_vis} AND g_ruwe.ruwe <{quality_par_ruwe}")
        # 2.- Run ADQL query ===================================
        warnings.simplefilter('ignore', category=AstropyWarning)
        print(f'RUNNING ADQL ASYNCRHRONOUS QUERY ' + '=' * 57)
        job       = Gaia.launch_job_async(query= query, verbose=True)
        self.cat  = job.get_results()

        flag_psn, flag_vis, flag_ruwe  = '', '', ''
        if quality_par_SN == '10':     flag_psn  = '(Default)'
        if quality_par_vis == '7':     flag_vis  = '(Default)'
        if quality_par_ruwe == '1.40': flag_ruwe = '(Default)'        

        if verbose:
            print('=' * 90)
            print(f'Selection Criteria in Parallax S/N:             Parallax S/N > {quality_par_SN}   {flag_psn}')
            print(f'Selection Criteria in Visibility Periods Used:  Vis          > {quality_par_vis}    {flag_vis}')
            print(f'Selection Criteria in RUWE:                     RUWE         < {quality_par_ruwe} {flag_ruwe}')
            print()            
            print(f'SAMPLE OUTPUT  N_els = {len(self.cat):3.0f}')
            print('=' * 90)
            print()


    def run_phot_crossmatch(self, upload_table = 'dummy.vot', verbose = True, add_input_cols = ""):
        """
        Run a crossmatch between Gaia, 2MASS and WISE using Gaia internal cross-matched tables.
        """
        # 1.- Search input table =============================== 
        upload_resource   = glob.glob(upload_table)
        upload_table_name = "input_table"
        if len(upload_resource) == 0:
            raise Exception('Upload Table not found')

        if add_input_cols != "":
            add_input_cols = "," + add_input_cols + " "

        # 2.- Write ADQL query =================================
        query  = ("SELECT " + self.gaia_cols_p + ", " + self.wise_cols + ", " + self.tmass_cols + " " + 
                add_input_cols + 
                "FROM tap_upload.input_table as input_table "
                "LEFT OUTER JOIN gaiadr2.gaia_source as gaia ON input_table.source_id = gaia.source_id "           # Inp Sample VS Gaia DR2
                "LEFT OUTER JOIN gaiadr2.allwise_best_neighbour as xmatch ON gaia.source_id = xmatch.source_id "   # Inp Sample VS WISE
                "LEFT OUTER JOIN gaiadr1.allwise_original_valid as wise ON xmatch.allwise_oid = wise.allwise_oid " # Inp Sample VS WISE          
                "LEFT OUTER JOIN gaiadr2.tmass_best_neighbour as xmatch_2 ON gaia.source_id = xmatch_2.source_id " # Inp Sample VS 2MASS
                "LEFT OUTER JOIN gaiadr1.tmass_original_valid as tmass ON xmatch_2.tmass_oid = tmass.tmass_oid "   # Inp Sample VS 2MASS
                f"WHERE cc_flags = '0000' "
                f"AND ext_flag < 2 "
                f"AND w3mpro_error IS NOT NULL "
                f"AND w4mpro_error IS NOT NULL")
        # 3.- Run ADQL query ===================================
        print(f'RUNNING ADQL SYNCRHRONOUS QUERY ' + '=' * 57)
        job      = Gaia.launch_job(query= query, upload_resource = upload_resource[0], upload_table_name = upload_table_name, verbose = True)
        self.cat = job.get_results()

        if verbose:
            print('=' * 90)
            print(f'SAMPLE OUTPUT  N_els = {len(self.cat):3.0f}')
            print('=' * 90)


    def run_gaia2mass_cross_match(self, upload_table = 'dummy.vot', verbose = True, add_input_cols = "", 
        quality_par_SN  = '10', quality_par_vis = '7',quality_par_ruwe = '1.40'):
        """
        Run a crossmatch between Gaia, 2MASS
        """
        # 1.- Search input table =============================== 
        upload_resource   = glob.glob(upload_table)
        upload_table_name = "input_table"
        if len(upload_resource) == 0:
            raise Exception('Upload Table not found')

        if add_input_cols != "":
            add_input_cols = "," + add_input_cols + " "


        # 2.- Write ADQL query ================================= 
        query  = ("SELECT input_table.col2mass, " + self.gaia_cols + 
                  ",sqrt(gaia.astrometric_chi2_al/(gaia.astrometric_n_good_obs_al-5)) as unit_weight_e, g_ruwe.ruwe as ruwe "
                  "FROM tap_upload.input_table as input_table "
                  "LEFT OUTER JOIN gaiadr2.tmass_best_neighbour AS xmatch ON input_table.col2mass = xmatch.original_ext_source_id "
                  "LEFT OUTER JOIN gaiadr2.gaia_source          AS gaia   ON xmatch.source_id     = gaia.source_id "
                  "LEFT OUTER JOIN gaiadr2.ruwe                 AS g_ruwe ON gaia.source_id       = g_ruwe.source_id "
                  f"WHERE gaia.parallax/gaia.parallax_error >{quality_par_SN} AND gaia.visibility_periods_used >{quality_par_vis} AND g_ruwe.ruwe <{quality_par_ruwe}")


        # 3.- Run ADQL query ===================================
        warnings.simplefilter('ignore', category=AstropyWarning)
        print(f'RUNNING ADQL SYNCRHRONOUS QUERY ' + '=' * 58)
        job       = Gaia.launch_job(query= query, upload_resource = upload_resource[0], upload_table_name = upload_table_name, verbose = verbose)
        self.cat  = job.get_results()

        flag_psn, flag_vis, flag_ruwe  = '', '', ''
        if quality_par_SN == '10':     flag_psn  = '(Default)'
        if quality_par_vis == '7':     flag_vis  = '(Default)'
        if quality_par_ruwe == '1.40': flag_ruwe = '(Default)'        

        if verbose:
            print('=' * 90)
            print(f'Selection Criteria in Parallax S/N:             Parallax S/N > {quality_par_SN}   {flag_psn}')
            print(f'Selection Criteria in Visibility Periods Used:  Vis          > {quality_par_vis}    {flag_vis}')
            print(f'Selection Criteria in RUWE:                     RUWE         < {quality_par_ruwe} {flag_ruwe}')
            print()            
            print(f'SAMPLE OUTPUT  N_els = {len(self.cat):3.0f}')
            print('=' * 63)                             