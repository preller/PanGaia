""""
Python Class with dedicated utilities/methods to analyse Gaia DR2 samples

Héctor Cánovas Oct 2019 - now
"""

import glob, warnings
import numpy as np
from astropy                  import units as u
from astropy.coordinates      import SkyCoord
from astropy.table            import Table, MaskedColumn
from astropy.utils.exceptions import AstropyWarning


class LibUtils():
    """
    Initialize the class.
    """
    def __init__(self, color = 'magenta', label = 'Sample'):
        self.color = color
        self.label = label
        self.bcols = ['ra', 'dec', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error', 'parallax','phot_g_mean_mag',
            'phot_bp_mean_mag', 'phot_rp_mean_mag', 'phot_g_mean_flux_over_error', 'phot_bp_mean_flux_over_error',
            'phot_rp_mean_flux_over_error'] # Basic Gaia cols

    
    def __repr__(self):
        return f'This Class contains tools to create and manipulate a Gaia catalogue'


    def make_cat(self, inp_tb):
        """
        Creates Catalogue (Astropy Table instance)
        """
        if isinstance(inp_tb, str):
            self.cat = Table.read(inp_tb, format = 'votable')
        if isinstance(inp_tb, Table):
            self.cat = inp_tb


    def read_catalogue(self, inp_tb = None, verbose = True, save_sample = False, print_vrad = False, 
            sample_dir = '../samples_control/'):
        """
        Read Gaia Sample.
        """
        warnings.filterwarnings('ignore', category=AstropyWarning, append=True)

        if inp_tb:
            self.make_cat(inp_tb)
        else:
            inp_cats = glob.glob(sample_dir + '*vot')
            if len(inp_cats) == 0:
                print(f'No catalogues found in {sample_dir}.')
            else:
                inp_cats = [inp_cat[inp_cat.rfind('/')+1:] for inp_cat in inp_cats]
                print(f'Sample Catalogues in {sample_dir}: ' + '='*46)
                print()
                for inp_cat in inp_cats:
                    print('* ' + inp_cat)
                print()    

                while True:
                    samp_con   = input('Choose Control Sample: ')
                    samp_con_i = glob.glob(sample_dir + samp_con)
                    if len(samp_con_i) == 1:
                        self.cat = Table.read(samp_con_i[0])
                        break
                    else:
                        print('Table not found; try again')
        if verbose:
                print(f'{self.label} loaded. N_elements (rows) = {len(self.cat)}')

        # Add columns/check catalogue ==============
        print()
        self.sanity_checker(verbose = verbose)
        self.to_cone_search(verbose = False)

        if print_vrad:
            self.get_vrad_stats()
        if save_sample:
            self.save_sample()


    def sanity_checker(self, verbose = True):
        """
        Make sure that input table is a Gaia catalogue and add new columns
        """
        for col in self.bcols:
            if col not in self.cat.colnames:
                print(f'Warning: {col} is missing in catalogue Table')
                raise Exception('Missing Column')
        if verbose:
            print('Checking catalogue columns (ra, dec, parallax, pmra, phot_g_mean_mag, etc) - OK')
        self.add_extra_cols(verbose = verbose)
        

    def add_extra_cols(self, verbose = True):
        """
        Add extra columnts to a Gaia DR2 sample.
        """
        if 'distance' not in self.cat.colnames:
            self.add_distance(verbose = verbose)
        if 'phot_g_mean_mag_abs' not in self.cat.colnames:
            self.add_absmag(verbose = verbose)
        if 'phot_g_mean_mag_err' not in self.cat.colnames:
            self.add_mag_errs(verbose = verbose)
        if 'l' not in self.cat.colnames:
            self.add_galactic(verbose = verbose)
        if 'X_gal' not in self.cat.colnames:
            self.add_3D_galactic(verbose = verbose)
        if 'X_gal' not in self.cat.colnames:
            self.add_3D_galactic(verbose = verbose)
        if 'pm_mod' not in self.cat.colnames: 
            self.add_pm_mod(verbose = verbose)


    def add_distance(self, verbose = True):
        """
        Add "distance" column to the catalogue, where distance = 1000./parallaxes.
        """
        ncol  = MaskedColumn(data = 1./self.cat['parallax'] * 1000, name = 'distance', unit = u.parsec, format = '4.1F')
        self.cat.add_column(ncol)
        if verbose:
            print('Adding new column to Gaia DR2 dataset: Distance')


    def add_absmag(self, verbose = True):
        """
        Add "absolute_magnitudes" columns to the catalogue.
        """
        for col in ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag']:
            self.cat[col + '_abs']        = self.cat[col] + (5. * np.log10(self.cat['parallax']*0.001) + 5) # Gaia Parallax in mas.
            self.cat[col + '_abs'].format = self.cat[col].format
            self.cat[col + '_abs'].unit   = self.cat[col].unit            
        if verbose:
            print('Adding new columns to Gaia DR2 dataset: Absolute Magnitudes')


    def add_mag_errs(self, verbose = True):
        """
        Compute photometric errors in magnitudes: mag_err ~ sigma_flux/flux #VALID ONLY FOR SMALL ERRORS
        # http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/stats/L18/index.html
        # https://www.eso.org/~ohainaut/ccd/sn.html        
        """
        gbands = ['g', 'bp', 'rp']
        for band in gbands:
            self.cat['phot_' + band + '_mean_mag_err']        = 1./self.cat['phot_' + band + '_mean_flux_over_error']
            self.cat['phot_' + band + '_mean_mag_err'].unit   = self.cat['phot_' + band + '_mean_mag'].unit
            self.cat['phot_' + band + '_mean_mag_err'].format = self.cat['phot_' + band + '_mean_mag'].format
        if verbose:
            print('Adding new columns to Gaia DR2 dataset: Magnitude Errors')


    def add_galactic(self, verbose = True):
        """
        Add "Galactic Coordinates" columns to the catalogue.
        """
        coords = SkyCoord(ra=self.cat['ra'], dec=self.cat['dec'], frame='icrs')
        self.cat['l'] = coords.galactic.l.degree * u.degree
        self.cat['b'] = coords.galactic.b.degree * u.degree

        self.cat['l'].format = '10.5F'
        self.cat['b'].format = '10.5F'
        if verbose:
            print('Adding new columns to Gaia DR2 dataset: Galactic Coordinates (l & b)')


    def add_3D_galactic(self, verbose = True):
        """
        Computes 3D cartesian coordinates in the Galactic frame
        """
        coords = SkyCoord(self.cat['l'], self.cat['b'], frame='galactic',  distance=self.cat['distance'])
        self.cat['X_gal'] = coords.cartesian.x
        self.cat['Y_gal'] = coords.cartesian.y
        self.cat['Z_gal'] = coords.cartesian.z
        if verbose:
            print('Adding new columns to Gaia DR2 dataset: Galactic Spatial Coordinates (X, Y, Z)_Gal')


    def add_pm_mod(self, verbose = True):
        """
        Compute the proper motion modulus
        """
        self.cat['pm_mod'] = np.sqrt(self.cat['pmra']**2 + self.cat['pmdec']**2)
        if verbose:
            print('Adding new columns to Gaia DR2 dataset: Proper Motion Modulus')


    def get_vrad_stats(self):
        """
        Print radial velocity information.
        """
        vrads = self.cat['radial_velocity'][self.cat['radial_velocity'].mask == False]
        print()
        print(f'vrad measurements for {len(vrads)} sources ({len(vrads)/len(self.cat) * 100:4.1F}% of the sample)')
        print(f'vrad = {vrads.mean():17.1F} +/- {vrads.std():3.1F} [{vrads.unit}]')

       
    def to_cone_search(self, verbose = False):
        """
        Computes average R.A./Dec coords, parallax range, and projected-sky size
        """
        ra             = self.cat['ra'].mean()  * self.cat['ra'].unit
        dec            = self.cat['dec'].mean() * self.cat['dec'].unit
        delta_ra       = np.abs(self.cat['ra'].max()  - self.cat['ra'].min())
        delta_dec      = np.abs(self.cat['dec'].max() - self.cat['dec'].min())
        radius         = np.max([delta_ra, delta_dec]) * 0.5  * self.cat['dec'].unit
        para_min       = np.floor(self.cat['parallax'].min()*100)/100. * self.cat['parallax'].unit
        para_max       = np.ceil(self.cat['parallax'].max()*100)/100.  * self.cat['parallax'].unit
        self.ADQL_pars = {'ra':ra, 'dec':dec, 'radius':radius, 'para_min':para_min, 'para_max':para_max}
        if verbose:
            print('sample properties saved for ADQL Cone-Search')

        
    def print_cone_properties(self):
        """
        Prints Sky-Properties of the Sample to prepare a Gaia Cone Search
        """
        if hasattr(self, 'ADQL_pars'):
            formatter = iter(['13.2F', '13.2F', '12.2F', '13.2F', '13.2F'])
            text      = iter(['Average R.A.', 'Average Dec.', 'Radius on-sky', 'Parallax min', 'Parallax max'])
            print(f'{self.label} on-Sky sample properties:')
            for key in self.ADQL_pars:
                print(f'* {next(text)} {self.ADQL_pars[key].value : {next(formatter)}}, {self.ADQL_pars[key].unit}')
        else:
            print('Sample has no ADQL parameters. Please run .to_cone_search()')


    def save_sample(self):
        """
        Save the catalogue as an Astropy Table
        """
        print()
        fname = self.label.replace(' ', '_') + '.vot'
        text  = f'Saving {self.label} as: {fname}'
        print('=' * len(text))
        print(text)
        print('=' * len(text))
        self.cat.write(fname, format = 'votable', overwrite = True)