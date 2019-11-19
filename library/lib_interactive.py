'''
Created on 26 August 2019

See also: 
https://github.com/jonmmease/plotly_ipywidget_notebooks/blob/master/notebooks/iris_dashboard.ipynb

@author: hcanovas
'''

import pandas   as pd
import numpy    as np
import copy
import plotly.graph_objs as go
from plotly.graph_objs import FigureWidget
from ipywidgets import HBox, VBox, Button
import webcolors


# Extra Methods ================================
def text_annote(text = 'dummy', x = 0., y = 0., font_size = 12, textangle = 0., showarrow = False, xref = 'paper', yref = 'paper'):
	"""
	Used by "Interactive.explore_and_select()"
	"""
	return go.layout.Annotation(text = text, x = x, y = y, showarrow = showarrow, font_size = font_size,
								xref = xref, yref = yref, textangle = textangle)

def color_nm_to_rgba(color = 'black', alpha = 0.5):
	ncolor = webcolors.name_to_rgb(color)
	return f'rgba({ncolor[0]},{ncolor[1]},{ncolor[2]},{alpha})'



class LibInteractive():
	"""
	Initialize the class.
	"""
	def __init__(self, color_all = 'black', color_high = 'cyan'):
		self.color_all   = color_all
		self.color_high  = color_high


	def __repr__(self):
		"""Return developer-friendly string representation."""
		return f'Class to interactively explor Gaia DR2 data. Input object must be a library_gaia.cat object catalogue'


	def load_gaia_cat(self, inp_cat):
		"""
		Load .vot table and export it to Pandas
		"""
		if inp_cat.__module__ == 'astropy.table.table':
			print('Exporting Astropy Table to Pandas')
			inp_cat = inp_cat.copy()
			inp_cat = inp_cat.to_pandas()
		self.cat = inp_cat


	def load_control_obj(self, control):
		"""
		Load control *object*. This includes catalogue, label, and color.
		"""
		self.control     = copy.deepcopy(control)
		if self.control.cat.__module__ == 'astropy.table.table':
			print('Exporting Control Table to Pandas')
			self.control.cat = self.control.cat.to_pandas()
		

	def set_marker(self, size = 8, border_width = 1, border_color = 'black', 
		show_scale = False, hist_marker = False, opacity = 1.0):
		"""
		Define generic Marker properties. It uses detault plotLy muted blue color (https://stackoverflow.com/questions/40673490/how-to-get-plotly-js-default-colors-list)
		"""
		color_def   = color_nm_to_rgba(color = self.color_all, alpha = 0.5)
		colorscale  = [[0, color_def], [0.5, color_def],[0.5, self.color_high], [1, self.color_high]]    # 2 colors for Default & Selected
		color       = np.zeros(len(self.cat))
		self.marker = {'cmax': 1.5, 'cmin': -0.5,'color': color,'colorscale': colorscale,'showscale': show_scale,
		               'line': dict(width = 1, color = 'black')} # Histogram layout is fixed

		if hist_marker == False:
			self.marker.update({'size':size, 'opacity':opacity, 'line':dict(width = border_width, color = border_color)})


	def set_marker_control(self): 
		"""
		Define properties of the control sample markers.
		"""
		self.marker_control = {'color': self.control.color, 'size': self.marker['size'] * 0.75, 'line': dict(width = 0)}


	def set_margins(self, fig, margins = None):
		"""
		Set Figure margins
		"""
		if margins:
			if 'l' in margins.keys(): fig.update_layout(margin = go.layout.Margin(l = margins['l']))
			if 'r' in margins.keys(): fig.update_layout(margin = go.layout.Margin(r = margins['r']))
			if 't' in margins.keys(): fig.update_layout(margin = go.layout.Margin(t = margins['t']))
			if 'b' in margins.keys(): fig.update_layout(margin = go.layout.Margin(b = margins['b']))
		return fig


	def set_fig_show(self, fig, margins = None, annotations = None, new_xrange = None, title = None):
		"""
		Set common Figure properties.
		NOTE: https://plot.ly/python/renderers/   >> "(...), so you should not use the .show figure method or the plotly.io.show function on FigureWidget objects."
		"""
		if new_xrange:
			fig.update_xaxes(range=new_xrange)
		if annotations:
			fig.update_layout(annotations = annotations)
		if title:
			if isinstance(title, str):
				title = dict(text = title, y = 0.9)
			fig.update_layout(title = go.layout.Title(text = title['text'], xref = 'paper', x = 0.5, y = title['y']))
		if margins: 
			fig = self.set_margins(fig, margins = margins)
		return fig


	def show_sky_coords(self, font_size = 16, width = 500, height = 500, y_xaxis = -0.15, x_yaxis = -0.2, title = None,
		margins = None, **kwargs):
		"""
		Interactive RA-DEC plot
		"""
		self.set_marker(**kwargs)

		fig = FigureWidget(**{
		    'data': [{'x': self.cat.ra,'y': self.cat.dec,'marker': self.marker,'mode': 'markers','type': 'scatter'}],
		    'layout': {'dragmode': 'pan', 'width': width, 'height': height, 'font_size':font_size} })

		xtit = text_annote(text = '$\\text{R.A [}^\circ]$', x = 0.5,    y = y_xaxis, font_size = font_size * 1.2)
		ytit = text_annote(text = '$\\text{Dec [}^\circ]$', x = x_yaxis, y = 0.5,    font_size = font_size * 1.2, textangle = -90)
		fig  = self.set_fig_show(fig, margins = margins, annotations = [xtit, ytit], title = title, new_xrange = [self.cat.ra.max(), self.cat.ra.min()])

		if hasattr(self, 'control'):
			self.set_marker_control()
			fig.add_trace(go.Scatter(x = self.control.cat.ra, y = self.control.cat.dec, marker = self.marker_control, mode = 'markers'))
			fig.update_layout(showlegend=False)
		self.fig_sky = fig


	def show_properm(self, font_size = 16, width = 500, height = 500, y_xaxis = -0.15, x_yaxis = -0.2, title = None,
		margins = None, **kwargs):
		"""
		Interactive Proper-Motions plot
		"""
		self.set_marker(**kwargs)

		fig = FigureWidget(**{
		    'data': [{'x': self.cat.pmra, 'y': self.cat.pmdec, 'marker': self.marker, 'mode': 'markers', 'type': 'scatter'}],
		    'layout': {'dragmode': 'pan', 'width': width, 'height': height, 'font_size':font_size} })

		xtit = text_annote(text = '$\\text{pmra [mas yr}^{-1}]$',  x = 0.5,     y = y_xaxis, font_size = font_size * 1.2)
		ytit = text_annote(text = '$\\text{pmdec [mas yr}^{-1}]$', x = x_yaxis, y = 0.5,     font_size = font_size * 1.2, textangle = -90)
		fig  = self.set_fig_show(fig, margins = margins, annotations = [xtit, ytit], title = title)
		if hasattr(self, 'control'):
			self.set_marker_control()
			fig.add_trace(go.Scatter(x = self.control.cat.pmra, y = self.control.cat.pmdec, marker = self.marker_control, mode = 'markers'))
			fig.update_layout(showlegend=False)
		self.fig_properm = fig


	def show_hist_dist(self, font_size = 16, width = 500, height = 500, y_xaxis = -0.15, x_yaxis = -0.2, title = None, 
		margins = None,**kwargs):
		"""
		Interactive Histogram (distances) plot
		"""
		self.set_marker(hist_marker = True, **kwargs)

		fig = FigureWidget(**{
		    'data': [{'x': self.cat.distance, 'marker': self.marker, 'type': 'histogram'}],
		    'layout': {'dragmode': 'pan', 'width': width, 'height': height, 'font_size':font_size} })

		xtit = text_annote(text='Distance [pc]',  x = 0.5,     y = y_xaxis, font_size = font_size * 1.2)
		ytit = text_annote(text='# Objects',      x = x_yaxis, y =  0.5,    font_size = font_size * 1.2, textangle = -90)
		fig  = self.set_fig_show(fig, margins = margins, annotations = [xtit, ytit], title = title)
		if hasattr(self, 'control'):
			marker_control_hist          = self.marker.copy()
			marker_control_hist['color'] = self.control.color
			fig.add_trace(go.Histogram(x = self.control.cat.distance, marker = marker_control_hist))
			fig.update_layout(barmode = 'overlay', showlegend=False)
		self.fig_hist_dist = fig


	def explore_and_select(self, font_size = 16, height = 500, marker_size = 8, **kargs):
		"""
		ALL-IN-ONE analysis window
		"""
		margins = dict(t = 80)
		width   = 500
		self.show_sky_coords(width = width, height = height, size = marker_size, margins = margins, y_xaxis = -0.15, title = dict(text = 'Sky Position',          y = 0.9), **kargs)
		self.show_properm(   width = width, height = height, size = marker_size, margins = margins, y_xaxis = -0.15, title = dict(text = 'Projected Velocity',    y = 0.9), **kargs)
		self.show_hist_dist( width = width, height = height, size = marker_size, margins = margins, y_xaxis = -0.15, title = dict(text = 'Distance Distribution', y = 0.9), x_yaxis = -0.15, **kargs)

		f1 = self.fig_sky
		f2 = self.fig_properm
		f3 = self.fig_hist_dist

		scatt1 = f1.data[0]
		scatt2 = f2.data[0]
		hist_3 = f3.data[0]

		# Interactive data selection =======================
		def brush(trace, points, state):
			inds = np.array(points.point_inds)
			if inds.size:
				self.cat_subsamp      = self.cat.loc[inds,:]       # Selected sample
				selected              = scatt1.marker.color.copy()
				selected[inds]        = 1
				scatt1.marker.color   = selected
				scatt2.marker.color   = selected

				#  Histograms require some care:
				# self.set_marker(hist_marker = True, color_def = color_high)
				self.set_marker(hist_marker = True)
				marker          = self.marker
				marker['color'] = self.color_high
				self.fig_hist_dist.add_trace(go.Histogram(x = self.cat.distance[inds], marker = self.marker))
				self.fig_hist_dist.update_layout(barmode = 'overlay', showlegend=False)

		scatt1.on_selection(brush)
		scatt2.on_selection(brush)
		# Reset Interactive Selection ======================
		def reset_brush(btn):
			selected            = np.zeros(len(self.cat))
			scatt1.marker.color = selected
			scatt2.marker.color = selected

			 # Remove sub-sample histogram
			if hasattr(self, 'control'): 
				f3.data = [f3.data[0], f3.data[1]]
			else:
				f3.data = [f3.data[0]]
			
			if hasattr(self, "cat_subsamp"):
				delattr(self, "cat_subsamp")

		# Create reset button ==============================		    
		button = Button(description="clear")
		button.on_click(reset_brush)

		# Widget ===============
		self.dashboard    = VBox([HBox([f1, f2,f3]), button])


	def show_3D_space(self, font_size = 16, width = 1200, height = 500, y_xaxis = -0.15, x_yaxis = -0.2, title = None,
		margins = None, **kwargs):
		"""
		Interactive Proper-Motions plot
		"""
		self.set_marker(**kwargs)
		
		fig = FigureWidget(**{
		    'data': [{'x': self.cat.X_gal, 'y': self.cat.Y_gal, 'z': self.cat.Z_gal, 'marker': self.marker, 'mode': 'markers', 'type': 'scatter3d'}],
		    'layout': {'dragmode': 'pan', 'width': width, 'height': height, 'font_size':font_size, 'showlegend': False}})

		if hasattr(self, 'control'):
			self.set_marker_control()
			fig.add_trace(go.Scatter3d(
				{'x': self.control.cat.X_gal, 'y': self.control.cat.Y_gal, 'z': self.control.cat.Z_gal, 'mode': 'markers', 'marker':self.marker_control}
				))

		if hasattr(self, 'cat_subsamp'):
			marker          = self.marker
			marker['color'] = self.color_high
			fig.add_trace(go.Scatter3d(
				{'x': self.cat_subsamp.X_gal, 'y': self.cat_subsamp.Y_gal, 'z': self.cat_subsamp.Z_gal, 'mode': 'markers', 'marker': marker}
				))
		
		fig.update_layout(scene = dict(xaxis_title = 'X_Gal [pc]', yaxis_title = 'Y_Gal [pc]', zaxis_title = 'Z_Gal [pc]'))
		fig = self.set_fig_show(fig, margins = margins, title = title)
		self.fig_3D = fig