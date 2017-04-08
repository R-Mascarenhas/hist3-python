# hist3-python
Plotting a 3D histogram in python

# Parameters:	
sample : array_like.		
The data to be histogrammed. It must be an (N,2) array or data that can be converted to such. The rows of the resulting array are the coordinates of points in a 2 dimensional polytope.

bins : sequence or int, optional, default: 10.
The bin specification:
A sequence of arrays describing the bin edges along each dimension.
The number of bins for each dimension (bins =[binx,biny])
The number of bins for all dimensions (bins = bins).

normed : bool, optional, default: False.
If False, returns the number of samples in each bin. If True, returns the bin density bin_count / sample_count / bin_volume.

color: string, matplotlib color arg, default = 'blue'

alpha: float, optional, default: 1.
0.0 transparent through 1.0 opaque

hold: boolean, optional, default: False

# Returns:	
H : ndarray.
The multidimensional histogram of sample x.

edges : list.
A list of 2 arrays describing the bin edges for each dimension.
