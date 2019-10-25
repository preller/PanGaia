'''
Created on 05 April 2019

Python Class to produce paper-quality plots from a Gaia DR2 sample.

@author: hcanovas
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy             import units as u
from astropy.stats       import histogram
from astropy.modeling    import models, fitting
from astropy.coordinates import SkyCoord
from astropy.table       import Table
from pyesasky.catalogue  import Catalogue
from pyesasky.cooFrame   import CooFrame

# ========================================================================================
# Gaia DR2 Plotters
# ========================================================================================
class LibPlotters():
	"""
	Initialize the class.
	"""
	def __init__(self):
		pass


	def __repr__(self):
		"""Return developer-friendly string representation."""
		return f'Class to plot processed Gaia DR2 data. Input object must be a Basic.cat catalogue'


	def load_gaia_cat(self, inp_cat):
		self.cat        = inp_cat


	def plot_3D(self):
		"""
		Plots 3D spatial (Cartesian X, Y, Z) distribution of the sample. 
		"""
		fig = plt.figure(figsize=[10,10])
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.cat['X'], self.cat['Y'], self.cat['Z'], c='k', marker='o')

		ax.set_xlabel('X [pc]')
		ax.set_ylabel('Y [pc]')
		ax.set_zlabel('Z [pc]')
		plt.show()


	def plot_2d(self, col_x = 'pmra', col_y = 'pmdec', fontsize = 18, fig = True, xlim = None, ylim = None, markersize = 2, 
	color = 'grey', alpha = 0.75, label = None, grid = True, fig_nm = None, mew = 0, markeredgecolor = 'black', 
	figsize = [7,7], show_av_err = False):
		"""
		Shows 2D plot. The plot is automatically centred in the average XY values.
		"""
		if fig:
			fig = plt.figure(figsize=figsize)
		if xlim:
			plt.xlim(xlim)
		if ylim:
			plt.ylim(ylim)

		plt.xlabel(col_x, fontsize = fontsize)
		plt.ylabel(col_y, fontsize = fontsize)
		plt.xticks(fontsize = fontsize)
		plt.yticks(fontsize = fontsize)
		plt.plot(self.cat[col_x], self.cat[col_y], 'o', markersize = markersize, mew = mew, markeredgecolor = 'black', color = color, alpha = alpha, label = label)
		ax       = plt.gca()
		xlims    = ax.get_xlim()
		ylims    = ax.get_ylim()
		xlims_rg = np.abs(np.max(xlims) - np.min(xlims))
		ylims_rg = np.abs(np.max(ylims) - np.min(ylims))
		if col_x == 'ra':
			ax.invert_xaxis()

		if show_av_err:
			xerr = np.median(self.cat[col_x + '_error'])/2.
			yerr = np.median(self.cat[col_y + '_error'])/2.
			x_e  = xlims[0] + xlims_rg/10
			y_e  = ylims[1] - ylims_rg/10
			ax.errorbar(x_e, y_e, xerr=xerr, yerr=yerr, fmt='o', color = 'black', markersize = markersize*0.1, capthick=2, capsize = markersize*0.2)

		if label: plt.legend(loc = 'upper right', fontsize = fontsize)
		if grid: plt.grid()		
		if fig: plt.show()
		if fig and fig_nm:
			fig.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
			print('=' * (len(fig_nm) + 14))
			print(f'PDF saved as: {fig_nm}')
			print('=' * (len(fig_nm) + 14))


	def oplot_2d(self, col_x = 'pmra', col_y = 'pmdec', markersize = 2, color = 'grey', alpha = 0.75, label = None, mew = 0, 
		markeredgecolor = 'black', fontsize = 18):
		"""
		OVERplot 2D plot
		"""
		plt.plot(self.cat[col_x], self.cat[col_y], 'o', markersize = markersize, mew = mew, markeredgecolor = 'black', color = color,
			alpha = alpha, label = label)
		if label: 
			plt.legend(loc = 'upper right', fontsize = fontsize)


	def plot_2d_and_hist(self, col_x = 'pmra', col_y = 'pmdec', col_hist = 'parallax', color_2d = 'grey', color_hist = 'lightgrey', 
		fontsize = 26, markersize = 12, label='Gaia', fig_nm = None, figsize = [25,12], **kargs):
		"""
		Combine in 1 plot 2D distribution (left quadrant) & Histogram (right quadrant)
		"""
		fig = plt.figure(figsize=figsize)

		plt.subplot(121)
		self.plot_2d(col_x = col_x, col_y = col_y, fontsize = fontsize, markersize = markersize, label=label,
			color = color_2d, mew = 0, fig = False, show_av_err = True, **kargs)

		plt.subplot(122)
		hist_g = self.plot_hist(col_hist, fontsize=fontsize, xtick_bins = 7, color_hist = color_hist, fig = False, show_ylabel = '# Sources')
		plt.show()

		# Save figure in .PDF ========================
		if fig_nm:
			fig.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
			print('=' * (len(fig_nm) + 14))
			print(f'PDF saved as: {fig_nm}')
			print('=' * (len(fig_nm) + 14))


	def plot_aitoff(self, x_coord = 'ra', y_coord = 'dec', markersize = 10, color = 'black', figsize = [15,7], title = None, fig_nm = None):
		"""
		Plots source distribution across the Celestial Sphere using Aitoff projection
		See: http://docs.astropy.org/en/stable/coordinates/skycoord.html
		"""
		coords  = SkyCoord(ra=self.cat[x_coord], dec=self.cat[y_coord], frame='icrs')
		ra_rad  = coords.ra.wrap_at(180 * u.deg).radian
		dec_rad = coords.dec.radian


		fig = plt.figure(figsize=figsize)
		plt.title(title)
		ax  = plt.subplot(111, projection="aitoff")
		ax.scatter(ra_rad, dec_rad, s = 20, c = 'black')
		plt.grid(True)
		plt.subplots_adjust(top=0.95,bottom=0.0)
		plt.show()
		if fig_nm:
			fig.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)


	# Histograms ====================================================
	def compute_hist(self, inp_col = 'parallax', hist_blocks = 'knuth'):
		"""
		Compute Histogram bins and heights
		"""
		inp_dat                  = self.cat[inp_col][self.cat[inp_col].mask == False]
		bin_heights, bin_borders = histogram(inp_dat, bins = hist_blocks)
		bin_center               = bin_borders[1:] - np.diff(bin_borders)[0]/2.
		self.hist_x              = bin_center
		self.hist_y              = bin_heights

	
	def plot_hist(self, inp_col = 'parallax', fontsize = 36, pad = 10, hist_blocks = 'knuth', fig = True, show_ylabel = None, xtick_bins = 5,
	xlim = None, ylim = None, color_hist = 'lightgrey', edgecolor='black', figsize=[7,5], fig_nm = None, **kargs):
		"""
		Plot Histogram
		"""
		if fig: 
			fig = plt.figure(figsize=figsize)
		if xlim: 
			plt.xlim(xlim)
		if ylim: 
			plt.ylim(ylim)
		if show_ylabel:
			plt.ylabel(show_ylabel, fontsize = fontsize)

		inp_col_label = inp_col
		if inp_col == 'parallax': inp_col_label = r'$\varpi$ [mas]'
		if inp_col == 'pmra':     inp_col_label = r'$\mu_{\alpha}^{*}$ [mas yr$^{-1}$]'
		if inp_col == 'pmdec':    inp_col_label = r'$\mu_{\delta}$ [mas yr$^{-1}$]'

		plt.xlabel(inp_col_label, fontsize = fontsize)
		plt.xticks(fontsize = fontsize)
		plt.yticks(fontsize = fontsize)
		plt.tick_params(axis='x', pad=pad)

		# Create Histogram ====
		self.compute_hist(inp_col = inp_col, hist_blocks = hist_blocks)
		plt.hist(self.hist_x, weights=self.hist_y, bins=len(self.hist_y), color = color_hist, edgecolor=edgecolor, **kargs)

		if xtick_bins != 0:
			plt.locator_params(axis='x', nbins=xtick_bins)
		if fig:
			plt.show()
		if fig_nm:
			fig.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
			print('=' * (len(fig_nm) + 14))
			print(f'PDF saved as: {fig_nm}')
			print('=' * (len(fig_nm) + 14))

		return {'bin_h': self.hist_y, 'bin_c': self.hist_x}


	def gaussfit_hist(self, show_plot = False, linewidth = 3):
		"""
		Fit Gaussian to Histogram bins & heights
		"""
		fitter          = fitting.LevMarLSQFitter()
		gaussian        = models.Gaussian1D(np.max(self.hist_y), np.mean(self.hist_x), np.std(self.hist_x))
		self.gfit       = fitter(gaussian, self.hist_x, self.hist_y)
		self.gfit_x     = np.arange(np.min(self.hist_x), np.max(self.hist_x), 0.01)
		self.gfit_y     = self.gfit(self.gfit_x)
		if show_plot:
			plt.plot(self.gfit_x, self.gfit_y, linewidth = linewidth)


	def plot_hist_and_gfit(self, inp_col = 'parallax', hist_blocks = 'knuth', **kargs):
		"""
		Plot Histogram + Gaussian Fit
		"""
		self.plot_hist(inp_col = inp_col,fig = False, **kargs)         # Compute & Plot Histogram
		self.gaussfit_hist(show_plot = True)                               # Apply Gaussian Fit to Histogram
		plt.show()


	def plot_3_hist(self, fig = True, inp_col_1 = 'parallax', inp_col_2 = 'pmra', inp_col_3 = 'pmdec', fig_nm = None, x1_bins = 0, x2_bins = 0, x3_bins = 0,
	 vl_1 = None, vl_2 = None, vl_3 = None, x1lim = None, x2lim = None, x3lim = None, l_color = 'black', l_width = 3, fontsize = 36, ylabel_1 = None, 
	 Gauss_fit = False):
		"""
		Plot 3 Histograms
		"""
		if fig:
			fig = plt.figure(figsize=[30,10])

		plt.subplot(131)
		hist_1  = self.plot_hist(inp_col_1, fig=False, show_ylabel = ylabel_1, xtick_bins = x1_bins, fontsize = fontsize)
		if Gauss_fit:
			self.gaussfit_hist(show_plot = True)
			hist_1['gfit'] = self.gfit
		if vl_1:
			plt.axvline(x=vl_1, linestyle = '--', color = l_color, linewidth = l_width)

		plt.subplot(132)
		hist_2  = self.plot_hist(inp_col_2, fig=False, xtick_bins = x2_bins, xlim = x2lim, fontsize = fontsize)
		if Gauss_fit:
			self.gaussfit_hist(show_plot = True)
			hist_2['gfit'] = self.gfit
		if vl_2:
			plt.axvline(x=vl_2, linestyle = '--', color = l_color, linewidth = l_width)

		plt.subplot(133)
		hist_3 = self.plot_hist(inp_col_3, fig=False, xtick_bins = x3_bins, xlim = x3lim, fontsize = fontsize)
		if Gauss_fit:
			self.gaussfit_hist(show_plot = True)
			hist_3['gfit'] = self.gfit
		if vl_3:
			plt.axvline(x=vl_3, linestyle = '--', color = l_color, linewidth = l_width)

		if fig:
			plt.show()
		if fig and fig_nm:
			fig.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
			print('=' * (len(fig_nm) + 14))
			print(f'PDF saved as: {fig_nm}')
			print('=' * (len(fig_nm) + 14))
		return {inp_col_1: hist_1, inp_col_2: hist_2, inp_col_3: hist_3}


	def send_to_ESASky(self, pyesasky_widget, background = 'WISE', color = 'white', catalogueName = 'Catalogue', radius = 1.0):
		"""
		Overplot Sample On-Sky using the pyESASky Jupyter Widget
		"""

		# Set EsasKy BackGround Color =========
		if background == 'WISE': back_dict = {'label':'AllWISE color', 'url': 'http://cdn.skies.esac.esa.int/AllWISEColor/'}

		pyesasky_widget.setHiPS(back_dict['label'], back_dict['url'])

		# Create Catalogue ====================
		catalogue = Catalogue(catalogueName = catalogueName, cooframe = CooFrame.FRAME_J2000, color = color, lineWidth = 1)
		for i in range(len(self.cat)):
		    catalogue.addSource(self.cat['source_id'][i], self.cat['ra'][i], self.cat['dec'][i], i + 1, [], [])
		pyesasky_widget.overlayCatalogueWithDetails(catalogue)

		# Set EsasKy FoV ======================
		pyesasky_widget.setGoToRADec(self.cat['ra'].mean(), self.cat['dec'].mean())
		pyesasky_widget.setFoV(radius * 2.0 * 1.5) # Increase FoV by 50% as it looks nicer


	def add_catalogue_to_ESASky(self, pyesasky_widget, new_catalogue, color = 'white', catalogueName = 'New_Catalogue', radius = None):
		new_catalogue_inp = Catalogue(catalogueName = catalogueName, cooframe = CooFrame.FRAME_J2000, color = color, lineWidth = 1)

		if isinstance(new_catalogue, pd.DataFrame):
			print('Exporting Pandas DF to .VOT')		
			new_catalogue = Table.from_pandas(new_catalogue)
		
		for i in range(len(new_catalogue)):
			new_catalogue_inp.addSource(new_catalogue['source_id'][i], new_catalogue['ra'][i], new_catalogue['dec'][i], i + 1, [], [])
		
		pyesasky_widget.overlayCatalogueWithDetails(new_catalogue_inp)
		pyesasky_widget.setGoToRADec(self.cat['ra'].mean(), self.cat['dec'].mean())
		if radius:
			pyesasky_widget.setFoV(radius)