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
from lib_utils       import LibUtils    as Utils
from lib_cluster     import LibCluster  as Cluster

class LibCompare():
    """
    Initialize the class.
    """
    def __init__(self):
        self.colors  = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Colors asigned to HDBSCAN Clusters


    def __repr__(self):
        """Return developer-friendly string representation."""
        return f'Class to create compare HDBSCAN output VS control sample'


    def read_control(self, control_obj, verbose = True, mew = 2):
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
                print(f'Cluster {j} contains {len(cluster):>5.0F} Elements, including {len(subs)} ({control_pc:4.1F}%) of the Control Sample')


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


    def plot_clusters_comp(self, alpha_main = 1.0, figsize = [30,9], markersize = 10, fontsize = 24, 
           ylim_3 = None, save_fig = True,  hist_blocks = 'knuth', mew = 1):
        """
        Plot clusters found by HDBSCAN - similar to library_cluster.plot_clusters()
        """
        # Load data to Plotter Class ======================
        self.labels = [f'Cluster {i}' for i in range(len(self.clusters))]


         # Clusters >> Utils objects =======
        cl_list = []
        for i in range(len(self.clusters)):
            cl_inp = Utils(color = self.colors[i], label = self.labels[i])
            cl_inp.read_catalogue(self.clusters[i], verbose = False)
            cl_list.append(cl_inp)

         # Load first cluster (largest) ====
        figs_cl0  = Plotters()
        figs_cl0.load_gaia_obj(cl_list[0])

         # Load the rest of the clusters ===
        figs_cls  = []
        for i in range(1,len(self.clusters)):
            figs   = Plotters()
            figs.load_gaia_obj(cl_list[i])
            figs_cls.append(figs)

         # Load control sample =============
        figs_ctl  = Plotters()
        figs_ctl.load_gaia_obj(self.control)

         # Merge together ==================
        # figs_cls = figs_cls + [figs_ctl]


        figure   = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0, wspace=0.25)
        # ================================================
        plt.subplot(131)
        figs_cl0.plot_2d(col_x = 'ra', col_y = 'dec', markersize = markersize, 
                  alpha = alpha_main, fontsize = fontsize, fig = False)

        for fig in figs_cls:
            fig.oplot_2d(col_x = 'ra', col_y = 'dec', markersize = markersize, alpha = 1, fontsize = fontsize, legend = True, mew = 1)
        figs_ctl.oplot_2d(col_x = 'ra', col_y = 'dec', markersize = markersize * 0.6, alpha = 1, fontsize = fontsize, legend = True, mew = 1)

        # ================================================
        plt.subplot(132)
        legend = False
        figs_cl0.plot_2d(col_x = 'pmra', col_y = 'pmdec', markersize = markersize, 
                  alpha = alpha_main, fontsize = fontsize, fig = False, legend = legend)

        for fig in figs_cls:
            fig.oplot_2d(col_x = 'pmra', col_y = 'pmdec', markersize = markersize, alpha = 1, fontsize = fontsize, legend = legend, mew = 1)
        figs_ctl.oplot_2d(col_x = 'pmra', col_y = 'pmdec', markersize = markersize * 0.6, alpha = 1, fontsize = fontsize, legend = legend, mew = 1)

        # ================================================
        plt.subplot(133)
        _ = figs_cl0.plot_hist(inp_col = 'distance', alpha = alpha_main, fontsize = fontsize, fig = False,
         hist_blocks = hist_blocks, ylim = ylim_3)

        for fig in figs_cls:
            _ = fig.plot_hist(inp_col = 'distance', alpha = 1, fontsize = fontsize, fig = False, 
                hist_blocks = hist_blocks, show_ylabel = '# Objects')
        _ = figs_ctl.plot_hist(inp_col = 'distance', alpha = 1, fontsize = fontsize, fig = False, 
            hist_blocks = hist_blocks, fill = False, hatch='//', linewidth=2)


        plt.show()
        if save_fig:
            fig_nm = f'{self.label}_hdb_minsamp_{self.min_samples}_prob_{self.probability}_.pdf'
            figure.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
            print('=' * (len(fig_nm) + 14))
            print(f'PDF saved as: {fig_nm}')
            print('=' * (len(fig_nm) + 14))