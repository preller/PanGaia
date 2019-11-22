""""
Python Class to apply HDBSCAN to a Gaia DR2 sample.

Héctor Cánovas May 2019 - now 
"""

import warnings
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization    import hist
from astropy.table            import Table
from sklearn                  import preprocessing
from lib_plotters             import LibPlotters as Plotters
from lib_utils                import LibUtils    as Utils

# Extra Methods ================================
def read_float_input(text = 'Introduce Input: '):
    while True:
        try:
            read_val = np.float(input(text))
            break
        except ValueError:
            print('This is not a number. Try again...')

    return read_val


class LibCluster():
    """
    Initialize the class.
    """
    def __init__(self, verbose = False):
        warnings.filterwarnings('ignore')
        self.colors  = plt.rcParams['axes.prop_cycle'].by_key()['color'] # HDBSCAN Cluster colors


    def __repr__(self):
        return f'Class to standarize data and apply HDBSCAN'


    def save_fig(self, figure, fig_nm = 'dummy.pdf', comment_len = 14, text = 'PDF saved as: '):
        """
        Save pdf & print info on screen.
        """
        figure.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
        print('=' * (len(fig_nm) + comment_len))
        print(f'{text}{fig_nm}')
        print('=' * (len(fig_nm) + comment_len))


    def save_tb(self, data_table, tb_nm = 'dummy.vot', text = '.vot table saved as: '):
        """
        Save .vot & print info on screen.
        """
        text   = f'{text}{tb_nm}'
        print('=' * (len(text)))
        print(text)
        print('=' * (len(text)))
        data_table.write(tb_nm, overwrite = True, format = 'votable')


    def save_clusters(self):
        """
        Save HDBSCAN clusters (final step)
        """
        i         = -1
        root_name = f'{self.label}_hdb_minsamp_{self.min_samples}_prob_{self.probability}_mCls_{self.mCls}'
        for cluster in self.clusters:
            i         = i + 1
            self.save_tb(cluster, tb_nm = root_name + f'_cl_{i}.vot', text = 'HDBSCAN Cluster data saved as: ')


    def load_gaia_obj(self, gaia_obj, scl_features = ['X_gal', 'Y_gal', 'Z_gal', 'pmdec', 'pmra'],
        scaler = None, save_scl = True):
        """
        Load Gaia Utils object.
        """
        if isinstance(gaia_obj, Utils):
            self.gaia_obj= gaia_obj
            self.data_tb = gaia_obj.cat
            self.label   = gaia_obj.label
            self.scl_data(scl_features = scl_features, scaler = scaler, save_scl = save_scl)
            self.plot_distributions()
        else:
            raise Exception('Input object is not a LibUtils instance')


    def scl_data(self, scl_features = ['X_gal', 'Y_gal', 'Z_gal', 'pmdec', 'pmra'], 
            scaler = None, verbose = False, save_scl = True):
        """
        Scale data (necessary step before applying any clustering algorithm)
        """
        if scaler == None:
            scaler = input('Introduce scaler option (standard (default), robust, minmax): ')
            if scaler == '': scaler = 'standard'
            if scaler != 'standard' and scaler != 'robust' and scaler != 'minmax': 
                print('wrong scaler; try again')

        self.scaler   = scaler
        self.features = scl_features 
        data_tb_pd    = self.data_tb[self.features].to_pandas()

        if self.scaler == 'standard':
            self.data_scl = preprocessing.StandardScaler().fit_transform(data_tb_pd)
        if self.scaler == 'minmax':
            self.data_scl = preprocessing.MinMaxScaler().fit_transform(data_tb_pd)
        if self.scaler == 'robust':
            self.data_scl = preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(data_tb_pd)

        if verbose:
            print('Printing Mean & Std Deviation of scaled data')
            scl_mean = iter(self.data_scl.mean(axis = 0))
            scl_std  = iter(self.data_scl.std(axis = 0))
            feats    = iter(self.features)
            for scl_mean_i in scl_mean:
                print(f'{next(feats):5s}: {scl_mean_i:>10.2f}, {next(scl_std):5.2f}')
        if save_scl:
            self.save_tb(Table(self.data_scl), tb_nm = f'{self.label}_scl_{self.scaler}.vot')


    def set_probability_thresold(self, probability = None, verbose = True):
        """
        Sources with membership probability below this thresold are excluded.
        """
        if probability == None:
            while True:
                try:
                    probability = input('Insert Probability-membership threshold (default = 70%): ')
                    if probability == '':
                        probability = 70.
                    else:
                        probability = np.float(probability)
                    break
                except ValueError:
                    print('Probability must be a number (Int/Float). Try again...')
            
        if probability > 1:
            probability = probability/100

        self.probability = probability
        if verbose:
            print(f'Probability thresold set to: {self.probability*100:4.0F}%')


    def set_min_samples(self, min_samples = -1, verbose = True):
        """
        Set Min Samples hyperparameter
        """
        if min_samples == -1:       
            while True:
                try:
                    min_samples = input('Insert min_samples value (default = None; 1, 2, 3...): ')
                    if min_samples == '':
                        min_samples = None
                    else:
                        min_samples = np.int(min_samples)
                    break
                except ValueError:
                    print('min_samples MUST be an integer. Try again...')

        self.min_samples = min_samples
        if verbose:
            if self.min_samples == None:
                print(f'minSamples set to: {"None":>16s}')
            else:
                print(f'minSamples set to: {self.min_samples:15.0F}')


    def run_hdbscan(self, min_cluster_size = None, verbose = False, probability = None, min_samples = -1, plot_clusters = False, **kargs):
        """
        Apply HDBSCAN algorithm. The code already applies a probability-membership selection criteria
        """
        self.set_probability_thresold(probability = probability, verbose = False)
        self.set_min_samples(min_samples = min_samples,          verbose = False)

        if min_cluster_size == None:
            while True:
                try:
                    min_cluster_size = input('Insert min_Cluster_size: ')
                    min_cluster_size = np.int(min_cluster_size)
                    break
                except ValueError:
                    print('min_cluster_size MUST be an integer >= 4 Try again...')

        min_cluster_size = np.int(min_cluster_size)
        if min_cluster_size < 4:
            raise Exception('min_cluster_size must be an integer >= 4.')

        self.mCls                   = min_cluster_size
        model                       = hdbscan.HDBSCAN(min_cluster_size = self.mCls, min_samples = self.min_samples)
        self.data_tb['label']       = model.fit_predict(self.data_scl)
        self.data_tb['Prob']        = model.probabilities_
        self.data_tb['Prob'].format = '3.2F'

        data_tb_hdb      = self.data_tb.copy()
        self.data_tb_hdb = data_tb_hdb[data_tb_hdb['Prob'] >= self.probability]
        self.clusters_extract(self.data_tb_hdb, verbose = verbose)

        if plot_clusters:
            self.plot_clusters(**kargs)


    def clusters_extract(self, inp_tab, verbose = True):
        """
        Extract the clusters identified by HDSBCAN. Input table must be a self.run_hdbscan result.
        """
        if len(inp_tab) >0:
            labels          = list(set(inp_tab['label']))
            clusters        = [inp_tab[inp_tab['label'] == label] for label in labels]
            # Sort clusters by size:
            lengths       = [len(inp) for inp in clusters]
            i_ordered     = np.argsort(np.array(lengths))  # Ordered indexes
            i_ordered     = i_ordered[::-1]                # Now from max to minimum size
            self.clusters = [clusters[inp] for inp in i_ordered]
        else:
            self.clusters = None
        self.clusters_get_info()
        if verbose:
            print(f'mCls = {self.mCls}; clusters = {self.clusters_n}; N_members = {self.clusters_l}')


    def clusters_get_info(self):
        """
        Get basic info about the cluster(s) found by HDBSCAN
        """
        if self.clusters:
            self.clusters_n = len(self.clusters)
            self.clusters_l = [len(inp) for inp in self.clusters] # Clusters N_members (lengths)
            self.clusters_r = [self.mCls] + self.clusters_l       # HDBSCAN resume. Useful for Multi-HDBSCAN
        else:
            self.clusters_n = 0
            self.clusters_l = 0
            self.clusters_r = 0


    def run_multi_hdbscan(self, mCls_min = 10, mCls_max = 70, mCls_step = 1, verbose = True, show_plot = False, 
            probability = None, min_samples = -1, **kargs):
        """
        Apply HDBSCAN algorithm to a range of mCls values defined by the user.
        """
        clusters_multi   = []
        clusters_multi_r = []
        print('')

        if verbose:
            print(f'Running HDBSCAN for mCls = {mCls_min}:{mCls_max} in steps of {mCls_step}')
            print()

        self.set_probability_thresold(probability = probability, verbose = True)
        self.set_min_samples(min_samples = min_samples,          verbose = True)
        print()

        for mCls in range(mCls_min, mCls_max, mCls_step):
            self.run_hdbscan(min_cluster_size = mCls, verbose = verbose, probability = self.probability,
                min_samples = self.min_samples, **kargs)

            if self.clusters_n > 0:
                clusters_multi.append(self.clusters)       # List of CLuster arrays
                clusters_multi_r.append(self.clusters_r)   # List of CLuster resumes

        self.multi_xrange     = [mCls_min, mCls_max]
        self.clusters_multi   = clusters_multi
        self.clusters_multi_r = clusters_multi_r
        if show_plot:
            self.plot_barchart(**kargs)


# Plotters ================================================================================================
    def plot_distributions(self, hist_blocks = 'knuth', color_hist  = 'lightgrey', edgecolor = 'black',
        fontsize = 16, save_fig = True):
        """
        Plot feature distributions (i.e., where the clustering is applied).
        Only works if data has been previously scaled.
        """
        index  = iter(np.arange(len(self.features)))
        figure = plt.figure(figsize=[30,10])
        
        for scl_col in self.features:
            index_i         = next(index)
            ax              = plt.subplot(2,len(self.features), 1 + index_i)
            bin_h, bin_b, _ = hist(self.data_tb[scl_col], hist_blocks, color = color_hist, edgecolor = edgecolor)
            plt.title(scl_col, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)

            ax              = plt.subplot(2, len(self.features),1 + len(self.features) + index_i)
            bin_h, bin_b, _ = hist(self.data_scl[:,index_i], hist_blocks, color = color_hist, edgecolor = edgecolor)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)

        plt.show()
        if save_fig:
            self.save_fig(figure, fig_nm = f'{self.label}_scl_{self.scaler}.pdf')


    def plot_clusters(self, alpha_main = 1.0, figsize = [30,9], markersize = 10, fontsize = 24, 
           ylim_3 = None, save_fig = True,  hist_blocks = 'knuth', mew = 1):
        """
        Plot clusters found by HDBSCAN
        """
        # Create labels ===================================
        self.labels = [f'Cluster {i}' for i in range(len(self.clusters))]

         # Clusters >> Utils objects =======
        cl_list   = []
        figs_list = []
        for i in range(len(self.clusters)):
            cl_inp = Utils(color = self.colors[i], label = self.labels[i])
            cl_inp.read_catalogue(self.clusters[i], verbose = False)
            cl_list.append(cl_inp)

         # Load first cluster (largest) ====
        figs_cl0  = Plotters()
        figs_cl0.load_gaia_obj(cl_list[0])

         # Load the rest of the clusters ===
        for i in range(1,len(self.clusters)):
            figs   = Plotters()
            figs.load_gaia_obj(cl_list[i])
            figs_list.append(figs)

        figure   = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0, wspace=0.25)
        # ================================================
        plt.subplot(131)
        figs_cl0.plot_2d(col_x = 'ra', col_y = 'dec', markersize = markersize, 
                  alpha = alpha_main, fontsize = fontsize, fig = False)
        for fig in figs_list:
            fig.oplot_2d(col_x = 'ra', col_y = 'dec', markersize = markersize, alpha = 1, 
                fontsize = fontsize, legend = True, mew = 1)
        # ================================================
        plt.subplot(132)
        legend = False
        figs_cl0.plot_2d(col_x = 'pmra', col_y = 'pmdec', markersize = markersize, 
                  alpha = alpha_main, fontsize = fontsize, fig = False, legend = legend)
        for fig in figs_list:
            fig.oplot_2d(col_x = 'pmra', col_y = 'pmdec', markersize = markersize, alpha = 1,
                fontsize = fontsize, legend = legend, mew = 1)
        # ================================================
        plt.subplot(133)
        _ = figs_cl0.plot_hist(inp_col = 'distance', alpha = alpha_main, fontsize = fontsize,
            fig = False, hist_blocks = hist_blocks, ylim = ylim_3)
        for fig in figs_list:
            _ = fig.plot_hist(inp_col = 'distance', alpha = 1, fontsize = fontsize, fig = False, 
                hist_blocks = hist_blocks, show_ylabel = '# Objects')
        plt.show()

        if save_fig:
            fig_nm = f'{self.label}_hdb_minsamp_{self.min_samples}_prob_{self.probability}_mCls_{self.mCls}.pdf'
            self.save_fig(figure, fig_nm = fig_nm)


    def plot_barchart(self, figsize = [15,7], fontsize = 18, save_fig = True):
        """
        Plot run_multi_hdbscan results using a bar-chart diagram.
        """
        # Create np.array for plot ===========
        x_dim         = np.max([len(inp) for inp in self.clusters_multi_r])
        y_dim         = len(self.clusters_multi)
        clusters_info = np.zeros([y_dim, x_dim])

        for y in range(y_dim):
            xrow = self.clusters_multi_r[y]
            if len(xrow) == x_dim:
                clusters_info[y] = xrow
            else:
                clusters_info[y][0:len(xrow)] = xrow
        
        self.clusters_multi_r_nparr = clusters_info
        # automatically adjust y_lim:
        y_max = [np.sum(inp) for inp in self.clusters_multi_r_nparr]
        ylim  = [0, np.max(y_max)]


        # Create Plot ========================
        figure   = plt.figure(figsize = figsize)
        bottoms  = [0]
        for xx in range(1,x_dim-1):
            bottoms.append(bottoms[xx-1] + clusters_info[:,xx])

        for i in range(x_dim-1):
            plt.bar(clusters_info[:,0], clusters_info[:,i+1], bottom = bottoms[i], label = f'Cluster {i+1}')

        plt.yticks(fontsize = fontsize)
        plt.xticks(fontsize = fontsize)
        plt.ylabel('# Cluster elements', fontsize = fontsize)
        plt.xlabel('min Cluster Size',   fontsize = fontsize)
        plt.legend(fontsize = fontsize*0.8)
        plt.xlim([self.multi_xrange[0]-5, self.multi_xrange[1]+5])
        plt.ylim(ylim)
        plt.show()

        if save_fig:
            fig_nm = f'{self.label}_hdb_minsamp_{self.min_samples}_prob_{self.probability}_barchart.pdf'
            self.save_fig(figure, fig_nm = fig_nm)


    def send_to_ESASky(self, pyesasky_widget, **kargs):
        """
        Show HDBSCAN clusters in ESASky
        """
        # Load data to Plotter Class ======================
        llabels    = iter([f'Cluster {i}' for i in range(self.clusters_n)])
        colors     = iter(self.colors)
        figs_data  = Plotters()
        i          = -1
        for cluster in self.clusters:
            i = i + 1
            cl_inp = Utils(color = next(colors), label = next(llabels))
            cl_inp.read_catalogue(cluster, verbose = False)
            figs_data.load_gaia_obj(cl_inp)
            figs_data.send_to_ESASky(pyesasky_widget, background = 'WISE', **kargs)            