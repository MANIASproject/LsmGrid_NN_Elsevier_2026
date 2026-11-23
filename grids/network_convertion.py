from grids import SpaceType
import sim_ballena as ballena

def convert_coords_to_2d( d2_coord,d3_coord, len_d3 ):
    assert d3_coord<len_d3, 'd3_coord cannot exceed len_d3'
    return d2_coord*len_d3 + d3_coord

def generate_network(grid, simulation_time, mode=SpaceType.D2, w=1.1, len_d3=None):
    if mode==SpaceType.D3 and len_d3 is None:
        raise Exception("If mode==3D, the len of the dim_3 is needed. Provide it! ")

    neuron_idx = 0
    reservoir  = []
    synases_in = []
    for cell in grid:
        t_start, t_end = cell.t_lims
        neu = ballena.Lif().v_thres(1).v_rest(0).v_rest(0).v_reset(0).tau(5*simulation_time).t_refractory(0).activity_window(t_start,t_end)
        reservoir.append( neu )

        if mode==SpaceType.D2:
            d2_start = int( cell.d2_lims[0] )
            d2_end   = int( cell.d2_lims[1] )
            for input_idx in range(d2_start,d2_end):
                synases_in.append( (input_idx,neuron_idx) )
        else:
            d2_start = int( cell.d2_lims[0] )
            d2_end   = int( cell.d2_lims[1] )
            d3_start = int( cell.d3_lims[0] )
            d3_end   = int( cell.d3_lims[1] )

            for d2_idx in range(d2_start, d2_end):
                for d3_idx in range(d3_start, d3_end):
                    converted_coord = convert_coords_to_2d(d2_idx, d3_idx, len_d3=len_d3)
                    synases_in.append( (converted_coord, neuron_idx) )
        neuron_idx += 1

    net = ( ballena.Network(reservoir)
        .synapses_in( synases_in )
        .synapses_net( [] )
        .weights( [w]*len(synases_in) )
        .outputs( [idx for idx in range(len(reservoir))] )
        .mode( ['SPIKES'] )
    )
    return net