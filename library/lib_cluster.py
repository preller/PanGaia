""""
Python Class to apply HDBSCAN to a Gaia DR2 sample.

Héctor Cánovas May 2019 - now 
"""

import warnings
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
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
        self.scalers = ['standard', 'minmax', 'robust']
        if verbose:
            print('Implemented scalers in this module:')
            for inp in self.scalers:
                print(inp)


    def __repr__(self):
        return f'Class to standarize data and apply HDBSCAN'


    def read_data_obj(self, data_obj, scl_features = ['X_gal', 'Y_gal', 'Z_gal', 'pmdec', 'pmra'], scaler = None):
        """
        Load Gaia Utils object
        """
        if isinstance(data_obj, Utils):
            self.data_tb = data_obj.cat
            self.label   = data_obj.label
            self.scl_data(scl_features = scl_features, scaler = scaler)
            self.save_data_scl(file_name = f'{self.label}_scl_{self.scaler}.vot')
            self.plot_distributions(file_name = f'{self.label}_scl_{self.scaler}.pdf')
        else:
            raise Exception('Input object is not a LibUtils instance')


    def read_data_tb(self, data_tb):
        """
        Load data (Astropy-Table object)
        """
        if isinstance(data_tb, str):
            self.data_tb = Table.read(data_tb, format = 'votable')
        if isinstance(data_tb, Table):
            self.data_tb = data_tb


    def read_data_scl(self, data_scl):
        """
        Read data (Astropy-Table object)
        """
        if isinstance(data_scl, str):
            data_scl      = Table.read(data_scl, format = 'votable')
            self.data_scl = data_scl.to_pandas()
        if isinstance(data_scl, Table):
            print('Exporting Astropy Table to Pandas')
            self.data_scl = data_scl.to_pandas()


    def scl_data(self, scl_features = ['X_gal', 'Y_gal', 'Z_gal', 'pmdec', 'pmra'], 
            scaler = None, verbose = False):
        """
        Scale data (necessary step before applying any clutering algorithm)
        """
        if scaler == None:
            scaler = input('Introduce scaler option (standard (default), robust, minmax): ')
            if scaler == '': scaler = 'standard'
            if scaler != 'standard' and scaler != 'robust' and scaler != 'minmax': 
                print('wrong scaler; try again')

        self.scaler   = scaler
        self.features = scl_features 
        data_tb_pd    = self.data_tb[self.features].to_pandas()

        if self.scaler == 'standard': self.data_scl = preprocessing.StandardScaler().fit_transform(data_tb_pd)
        if self.scaler == 'minmax':   self.data_scl = preprocessing.MinMaxScaler().fit_transform(data_tb_pd)
        if self.scaler == 'robust':   self.data_scl = preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(data_tb_pd)

        if verbose:
            print('Printing Mean & Std Deviation of scaled data')
            scl_mean = iter(self.data_scl.mean(axis = 0))
            scl_std  = iter(self.data_scl.std(axis = 0))
            feats    = iter(self.features)
            for scl_mean_i in scl_mean:
                print(f'{next(feats):5s}: {scl_mean_i:>10.2f}, {next(scl_std):5.2f}')


    def save_data_scl(self, file_name = 'data_scaled.vot'):
        """
        Save scaled data as Astropy table
        """
        text   = f'Scaled data saved as: {file_name}' 
        print('=' * (len(text)))
        print(text)
        print('=' * (len(text)))
        data_tb = Table(self.data_scl)
        data_tb.write(file_name, overwrite = True, format = 'votable')


    def save_cluster(self):
        """
        Save HDBSCAN selected cluster
        """
        index     = read_float_input('Select cluster index to save: ')
        index     = np.int(index)
        file_name = f'{self.label}_minsamp_{self.min_samples}_mCls_{self.mCls}_cl_{index}.vot'
        text   = f'HDBSCAN Cluster data saved as: {file_name}' 
        print('=' * (len(text)))
        print(text)
        print('=' * (len(text)))
        self.clusters[index].write(file_name, overwrite = True, format = 'votable')


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
                print(f'minSamples set to: None')
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
            self.plot_hdbscan_clusters(**kargs)


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
            self.clusters_get_info()
            if verbose:
                print(f'mCls = {self.mCls}; clusters = {self.clusters_n}; N_members = {self.clusters_l}')
        else:
            self.clusters = None
            self.clusters_get_info()


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


    def run_multi_hdbscan(self, mCls_min = 10, mCls_max = 70, mCls_step = 1, verbose = True, show_plot = True, 
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
            self.plot_multi_hdbscan_stats(**kargs)


# Plotters ================================================================================================
    def get_feature_lims(self, feature = 'pmra'):
        """
        Finds MAx/Min values for a given feature across the clusters found by HDSBCAN.
        This is useful to define the xylim values of the plots.
        """
        xymax = np.ceil(np.max([np.max(inp[feature])  for inp in self.clusters]))
        xymin = np.floor(np.min([np.min(inp[feature])  for inp in self.clusters]))

        return [xymin, xymax]


    def plot_distributions(self, hist_blocks = 'knuth', color_hist  = 'lightgrey', edgecolor = 'black',
        fontsize = 16, file_name = None):
        """
        Plot feature distributions (i.e., where the clustering is applied).
        Only works if data has been previously scaled.
        """
        scl_cols    = self.features
        index       = iter(np.arange(len(scl_cols)))

        fig = plt.figure(figsize=[30,10])
        
        for scl_col in scl_cols:
            index_i = next(index)
            ax              = plt.subplot(2,len(scl_cols), 1 + index_i)
            bin_h, bin_b, _ = hist(self.data_tb[scl_col], hist_blocks, color = color_hist, edgecolor = edgecolor)
            plt.title(scl_col, fontsize = fontsize)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)

            ax              = plt.subplot(2, len(scl_cols),1 + len(scl_cols) + index_i)
            bin_h, bin_b, _ = hist(self.data_scl[:,index_i], hist_blocks, color = color_hist, edgecolor = edgecolor)
            plt.xticks(fontsize = fontsize)
            plt.yticks(fontsize = fontsize)

        plt.show()
        if file_name:
            fig.savefig(file_name, bbox_inches = 'tight', overwrite = True)
            text = f'PDF saved as: {file_name}' 
            print('=' * (len(text)))
            print(text)
            print('=' * (len(text)))


    def plot_hdbscan_clusters(self, color_main = 'grey', alpha_main = 0.5, figsize = [30,9], markersize = 10, fontsize = 24,
        ylim_3 = None, fig_nm = None,  hist_blocks = 'knuth'):
        """
        Plot clusters found by HDBSCAN
        """
        # Load data to Plotter Class ======================
        figs_data  = Plotters()
        figs_data.load_gaia_cat(self.data_tb)
        color_def  = plt.rcParams['axes.prop_cycle'].by_key()['color']
        figs_cls   = []
        llabels    = iter([f'Cluster {i}' for i in range(self.clusters_n)])
        colors     = cycle(color_def) # Default MatPlotlub colors

        if self.clusters:
            for inp in self.clusters:
                fclass       = Plotters()
                fclass.color = next(colors)
                fclass.load_gaia_cat(inp)
                figs_cls.append(fclass)

        # ================================================
        figure   = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0, wspace=0.25)
        plt.subplot(131)
        col_x, col_y  = 'ra', 'dec'
        figs_data.plot_2d(col_x = col_x, col_y = col_y, markersize = markersize, color = color_main, 
                          alpha = alpha_main, fontsize = fontsize, fig = False, label = 'Data')
        plt.legend(fontsize = fontsize * 0.9)

        for fig in figs_cls:
            fig.oplot_2d(col_x = col_x, col_y = col_y, markersize = markersize, alpha = 1, label = next(llabels), color = fig.color)

        # ================================================
        plt.subplot(132)
        col_x, col_y  = 'pmra', 'pmdec'
        figs_data.plot_2d(col_x = col_x, col_y = col_y, markersize = markersize, color = color_main,
                          alpha = alpha_main, fontsize = fontsize, fig = False, xlim = self.get_feature_lims(col_x), 
                          ylim = self.get_feature_lims(col_y))
        for fig in figs_cls:
            fig.oplot_2d(col_x = col_x, col_y = col_y, markersize = markersize, alpha = 1, color = fig.color)

        # ================================================
        plt.subplot(133)
        inp_col = 'distance'
        _ = figs_data.plot_hist(inp_col = inp_col, color_hist = color_main, alpha = alpha_main, fontsize = fontsize, fig = False,
         hist_blocks = hist_blocks, ylim = ylim_3)

        for fig in figs_cls:
            _ = fig.plot_hist(inp_col = inp_col, color_hist = fig.color, alpha = 1, fontsize = fontsize, fig = False, 
                hist_blocks = hist_blocks, show_ylabel = '# Objects')

        plt.show()
        if fig_nm:
            figure.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
            print('=' * (len(fig_nm) + 14))
            print(f'PDF saved as: {fig_nm}')
            print('=' * (len(fig_nm) + 14))


    def esasky_hdbscan_clusters(self, pyesasky_widget, **kargs):
        """
        Show HDBSCAN clusters in ESASky
        """
        # Load data to Plotter Class ======================
        figs_data  = Plotters()
        i          = -1
        color_def  = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for cluster in self.clusters:
            i = i + 1
            figs_data.load_gaia_cat(cluster)
            figs_data.send_to_ESASky(pyesasky_widget, background='WISE', color=next(color_def), catalogueName = f'Cluster {i}', **kargs)


    def plot_multi_hdbscan_stats(self, figsize = [15,7], fontsize = 18, fig_nm = None):
        """
        Plot run_multi_hdbscan results using a bar-chart diagram.
        """
        # Create np.array for plot ===========
        x_dim         = np.max([len(inp) for inp in self.clusters_multi_r]) #Extra col contains mCls
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
        fig      = plt.figure(figsize = figsize)
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

        if fig_nm:
            if fig_nm == 'default':
                fig_nm = f'{self.label}_hdb_minsamp_{self.min_samples}_prob_{self.probability}.pdf'

            fig.savefig(fig_nm, bbox_inches = 'tight', overwrite = True)
            print('=' * (len(fig_nm) + 14))
            print(f'PDF saved as: {fig_nm}')
            print('=' * (len(fig_nm) + 14))