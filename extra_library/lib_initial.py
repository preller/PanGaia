""""
Dedicated module to create an initial sample from a catalogue containing targets coordinates or SIMBAD names.

Héctor Cánovas May 2019 - now
"""

from astropy              import units as u
from astropy.coordinates  import SkyCoord
from astropy.table        import Table, vstack
from astroquery.vizier    import Vizier


vizier   = Vizier(columns=["*", "+_r"]) # Nearest Object is the first one of the table



def query_ids_0(inp_id, catalog = "II/246", radius = 1.0 * u.arcsecond, verbose = False):
    """"
    Level-0 code to run individual ID queries in VizieR"
    """
    if verbose:
        print('Querying Vizier, ID: ',inp_id)

    viz_0  = vizier.query_object(inp_id,catalog=catalog, radius = radius)
    if len(viz_0) == 0:
        out_tb           = None
    else:
        out_tb           = viz_0[0]
        out_tb['dummy']  = 'N'
        out_tb['inp_id'] = inp_id
    return out_tb


def query_ids_1(inp_ids, catalog = "II/246", radius = 1.0 * u.arcsecond, verbose = True):
    """"
    Level-1 code to perform multiple ID queries in VizieR"
    """
    # 1) Construct Dummy Row ===========
    Vizier.ROW_LIMIT  = 1
    cat_0             = Vizier.get_catalogs(catalog)
    dummy_tb          = cat_0[0]
    dummy_tb['_r']    = 0
    dummy_tb['dummy'] = 'Y'
    Vizier.ROW_LIMIT  = -1    

    # 2) Construct Seed Table ==========
    viz_tb = dummy_tb.copy()

    # 3) Add rows to Seed Table ========
    for inp in inp_ids:
        result = sample_initial.query_ids_0(inp,catalog=catalog, radius = radius, verbose = verbose)
        viz_tb = vstack([viz_tb, Table(result)])

    viz_tb       = viz_tb[viz_tb['dummy'] != 'Y']
    viz_tb['_r'] = viz_tb['_r'].to(u.arcsecond)
    viz_tb['_r'].format = '3.2f'
    viz_tb.sort('inp_id')

    return viz_tb


def query_coords(inp_coords, catalog = "II/246", verbose = False, radius = 1.0*u.arcsecond):
    """
    Level-1 code to perform individual & multiple coordinate queries in VizieR"
    """
    out_0  = vizier.query_region(inp_coords[0],catalog = catalog, radius = radius)
    out_0  = out_0[0] # Get First table
    out_00 = Table([[0] for inp in out_0.colnames], names = out_0.colnames)

    out_0['inp_id'] = 0
    inp_ids         = iter(range(1,len(inp_coords)))

    for inp in inp_ids:
        if verbose:
            print('Querying Vizier, ID: ',inp)

        result = vizier.query_region(inp_coords[inp],catalog = catalog, radius = radius)
        if len(result) > 0:
            out = result[0]
        else:
            out = out_00

        out['inp_id'] = inp
        row           = out[0]
        out_0.add_row(row)

    out_0['dist']        = out_0['_r'].to(u.arcsecond)
    out_0['dist'].format = '3.2f'
    out_0.remove_column('_r')
    return out_0


def rename_2mass(inp_tb, s_cols = ['RAJ2000', 'DEJ2000', 'col2mass', 'jmag']):
    """
    Read 2MASS output. Rename columns fo Gaia DR2 archive input"
    """
    inp_tb.convert_bytestring_to_unicode()
    inp_tb.rename_column('_2MASS', 'col2mass') # For Gaia Archive
    inp_tb.rename_column('Jmag', 'jmag')       # For Gaia Archive
    inp_tb = inp_tb[s_cols]
    inp_tb['Name_id'] = ['2MASS J' + np.str(inp) for inp in inp_tb['col2mass']]

    return inp_tb