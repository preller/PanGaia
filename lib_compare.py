""""
Dedicated Class to compare the Control Sample VS. HDBSCAN selected clusters

Héctor Cánovas May 2019 - now
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools       import cycle
from astropy         import units as u
from astropy.table   import Table
from lib_plotters    import LibPlotters as Plotters
from lib_utils       import LibUtils       as Utils
from lib_cluster     import LibCluster     as Cluster

class LibCompare():
    """
    Initialize the class.
    """
    def __init__(self):
        pass


    def __repr__(self):
        """Return developer-friendly string representation."""
        return f'Class to create compare HDBSCAN output VS control sample'


    def read_control(self, control_obj, verbose = True, mew = 1.5):
        """
        Read Control Sample from control object
        """
        if isinstance(control_obj, Utils):
            self.control     = control_obj
            self.control.mew = mew
        else:
            raise Exception('Input is not a LibUtils object')


    def read_clusters(self, dbscan_obj, verbose = True):
        """
        Load HDBSCAN output clusters
        """
        if isinstance(dbscan_obj, Cluster):
            self.clusters    = dbscan_obj.clusters
            self.label       = dbscan_obj.label
            self.min_samples = dbscan_obj.min_samples
            self.probability = dbscan_obj.probability
            if verbose:
                i = 0
                for cluster in self.clusters:
                    i = i + 1
                    print(f'Cluster {i} contains {len(cluster)} Elements')
        else:
            raise Exception('Input is not a LibCluster object')


    def compare_to_control(self, verbose = True):
        """
        Compare HDBSCAN clusters to Control Sample.
        """
        if verbose:
            print(f'Comparing HDBSCAN clusters to Control Sample:')
        j = -1
        for cluster in self.clusters:
            j = j +1 
            cluster['Control'] = ['N'] * len(cluster)

            for i in range(len(cluster)):
                if cluster['source_id'][i] in self.control.cat['source_id']:
                    cluster['Control'][i] = 'Y'

            subs       = cluster[cluster['Control'] == 'Y']
            control_pc = (len(subs)/len(self.control.cat) * 100)

            self.clusters[j]['Control'] = cluster['Control']

            if verbose:
                print(f'Cluster {j} contains {len(cluster):>5.0F} Elements, \
                    including {len(subs)} ({control_pc:4.1F}%) of the Control Sample')


    def get_new_from_cl(self, cl_index = None, write_simbad_query = True, verbose = True):
        """
        Extract cluster members that do not belong to the Control Sample (New member Candidates)
        """
        
        if cl_index == None:    
            i = -1
            for cluster in self.clusters:
                i = i + 1
                print(f'Cluster {i}    {i}')
            cl_index = input('Choose Cluster from list: ')
            cl_index = np.int(cl_index)

        cluster  = self.clusters[cl_index]
        self.new = cluster[cluster['Control'] == 'N']
        self.new['Simbad'] = [f'Gaia DR2 {np.str(inp)}' for inp in self.new['source_id']]
        
        fname = 'simbad_list.txt'
        if write_simbad_query:
            Table([self.new['Simbad']]).write(fname, format = 'ascii.fast_no_header', overwrite = True)
        if verbose:
            text   = f'List of new member candidates for Simbad query saved as: {fname}'
            print('=' * len(text))
            print(text)
            print('=' * len(text))


    def plot_clusters_comp(self, figsize = [30,9], markersize = 10, fontsize = 24, 
            xlim_1 = None, ylim_1 = None, xlim_2 = None, ylim_2 = None, ylim_3 = None,
            fig_nm = None,  hist_blocks = 'knuth', mew = 1):
        """
        Plot clusters found by HDBSCAN - similar to library_cluster.plot_hdbscan_clusters()
        """
        # Load data to Plotter Class ======================
        figs_data  = Plotters()
        figs_data.load_gaia_cat(self.clusters[0]) # Load first the largest cluster (first on the list)
        color_def  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        figs_cls   = []
        llabels    = iter([f'Cluster {i+1}' for i in range(len(self.clusters))])
        color_0    = color_def[0]
        colors     = cycle(color_def[1:]) # Default MatPlotlub colors
        control_sf = 0.6
        
        # Load Control data =====
        fig_con    = Plotters()
        fig_con.load_gaia_cat(self.control.cat)

        for inp in self.clusters[1:]:
            fclass       = Plotters()
            fclass.color = next(colors)
            fclass.load_gaia_cat(inp)
            figs_cls.append(fclass)

        # ================================================
        figure   = plt.figure(figsize=figsize)
        plt.subplot(131)
        col_x, col_y = 'ra', 'dec'
        
        figs_data.plot_2d(col_x = col_x, col_y = col_y, markersize = markersize, 
            color = color_0, fontsize = fontsize, fig = False, label = 'Cluster 0',
            xlim = xlim_1, ylim = ylim_1, mew = mew)
        plt.legend(fontsize = fontsize * 0.9)

        for fig in figs_cls:
            fig.oplot_2d(col_x = col_x, col_y = col_y, markersize = markersize,
                label = next(llabels), color = fig.color, mew = mew)

        fig_con.oplot_2d(col_x = col_x, col_y = col_y, markersize = markersize*control_sf,
            label = self.control.label, color = self.control.color, mew = self.control.mew)


        # ================================================
        plt.subplot(132)
        col_x, col_y = 'pmra', 'pmdec'
        figs_data.plot_2d(col_x = col_x, col_y = col_y, markersize = markersize, color = color_0,
                          fontsize = fontsize, fig = False, xlim = xlim_2, ylim = ylim_2, mew = mew)
        for fig in figs_cls:
            fig.oplot_2d(col_x = col_x, col_y = col_y, markersize = markersize, color = fig.color, mew = mew)

        fig_con.oplot_2d(col_x = col_x, col_y = col_y, markersize = markersize*control_sf,
            color = self.control.color, mew = self.control.mew)


        # ================================================
        plt.subplot(133)
        inp_col = 'distance'
        _ = fig_con.plot_hist(inp_col = inp_col, color_hist = self.control.color, fontsize = fontsize,
                fig = False, hist_blocks = hist_blocks)

        _ = figs_data.plot_hist(inp_col = inp_col, color_hist = color_0, fontsize = fontsize,
                fig = False, hist_blocks = hist_blocks, ylim = ylim_3)

        for fig in figs_cls:
            _ = fig.plot_hist(inp_col = inp_col, color_hist = fig.color, fontsize = fontsize,
                    fig = False, hist_blocks = hist_blocks)

        
        plt.show()
        if fig_nm:
            if fig_nm == 'default':
                fig_nm = f'{self.label}_hdb_minsamp_{self.min_samples}_prob_{self.probability}_.pdf'
            figure.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
            print('=' * (len(fig_nm) + 14))
            print(f'PDF saved as: {fig_nm}')
            print('=' * (len(fig_nm) + 14))
