""""
Python Class with dedicated methods to create a control sample.

Héctor Cánovas Oct 2019 - now
"""

import glob, warnings
import numpy as np
from astropy.table            import Table, MaskedColumn
from astropy.utils.exceptions import AstropyWarning


class LibControl():
    """
    Initialize the class.
    """
    def __init__(self, color = 'magenta', label = 'Sample_Control'):
        self.color = color
        self.label = label

    
    def __repr__(self):
        return f'This Class contains tools to create and manipulate a Control Sample from a Gaia catalogue'


    def load_cat(self, inp_tb):
        """
        Creates Catalogue (Astropy Table instance)
        """
        if isinstance(inp_tb, Table):
            self.cat = inp_tb
        else:
            print('Input table must be an Astropy/Table instance')



    def remove_hist_outliers(self, histograms_dat, N_sigma = None, verbose = False):
        """
        It removes outliers in Distance, PMRA, and PMDEC histograms. 
        * histograms_dat >> Output from library_plotters.plot_3_hist(Gauss_fit = True)
        """
        if N_sigma == None:
            while True:
                try:
                    N_sigma = np.float(input('Introduce N_Sigma threshold: '))
                    break
                except ValueError:
                    print('N_sigma must be a number (Int/Float). Try again...')

        hist_cols = list(histograms_dat.keys())

        for hist_col in hist_cols:
            histo  = histograms_dat[hist_col]
            hrange = np.abs(histo['bin_c'].max() - histo['bin_c'].min())
            inner  = histo['gfit'].mean - histo['gfit'].stddev*(N_sigma)
            outer  = histo['gfit'].mean + histo['gfit'].stddev*(N_sigma)
            grange = np.abs(outer - inner)

            if hrange < grange:
                print(f'{hist_col:<10s}: Range too narrow to remove outliers')
            else:
                self.apply_col_cut(inp_col=hist_col, inner = inner, outer = outer)
                if verbose:
                    print(f'Removing outliers in: {hist_col}')

        text = f'Removing targets at {N_sigma} sigma from Gaussian Centers'
        print('=' * len(text))
        print(text)
        print(f'Targets in Control Sample: {len(self.cat):>20.0f}')
        print('=' * len(text))


    def apply_col_cut(self, inp_col, inner = 0, outer = 0):
        """
        It removes Table elements that do not satisfy the boundary conditions for the selected column.
        """
        els = (self.cat[inp_col] > inner) & (self.cat[inp_col] < outer)
        self.cat = self.cat[els]


    def save_control_sample(self):
        """
        Save the catalogue in an Astropy Table
        """
        print()
        text = f'Saving Control Sample as: {self.label + ".vot"}'
        print('=' * len(text))
        print(text)
        print('=' * len(text))
        self.cat.write(self.label + '.vot', format = 'votable', overwrite = True)