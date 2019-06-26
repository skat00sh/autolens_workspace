from autolens.data import ccd
from autolens.data import simulated_ccd
from autolens.data.array import grids
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.lens import lens_fit
from autolens.lens import lens_data as ld
from autolens.lens import plane as pl
from autolens.model.inversion import inversions as inv
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.lens.plotters import lens_fit_plotters

# In this tutorial, we'll introduce a new pixelization, called an adaptive-pixelization. This pixelization doesn't use
# uniform grid of rectangular pixels, but instead uses irregular 'Voronoi' pixels. So, why would we want to do that?
# Lets take another look at the rectangular grid, and think about its weakness.

# This is the image we're used to fitting with inversions now.
def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(shape=(150, 150), pixel_scale=0.05,
                                                                       psf_shape=(11, 11))

    lens_galaxy = g.Galaxy(redshift=0.5,
                           mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                        einstein_radius=1.6))
    source_galaxy = g.Galaxy(redshift=1.0,
                             light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.7, phi=135.0, intensity=0.2,
                                                       effective_radius=0.2, sersic_index=2.5))
    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.05,
        exposure_time=300.0, psf=psf, background_sky_level=1.0, add_noise=True), tracer.profile_image_plane_image_2d

# Lets quickly remind ourselves of the image, and the 3.0" circular mask we'll use to mask it.
ccd_data, hyper_image_2d = simulate()
mask = msk.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)
hyper_image = mask.map_2d_array_to_masked_1d_array(array_2d=hyper_image_2d)
# ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask)

# The lines of code below do everything we're used to, that is, setup an image and its grid stack, mask it, trace it
# via a tracer, setup the rectangular mapper, etc.
lens_galaxy = g.Galaxy(redshift=0.5,
                       mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                    einstein_radius=1.6))

lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=4)

adaptive = pix.VoronoiBrightnessImage(pixels=300.0, weight_floor=0.02, weight_power=5.0)

cluster_weight_map = adaptive.cluster_weight_map_from_hyper_image(hyper_image=hyper_image)

sparse_to_regular_grid = grids.SparseToRegularGrid.from_total_pixels_cluster_grid_and_cluster_weight_map(
    total_pixels=adaptive.pixels, cluster_grid=lens_data.grid_stack.regular, cluster_weight_map=cluster_weight_map, seed=1)

image_plane_grid_stack = lens_data.grid_stack.new_grid_stack_with_grids_added(
    pixelization_grid=sparse_to_regular_grid.sparse, regular_to_pixelization=sparse_to_regular_grid.regular_to_sparse)

image_plane = pl.Plane(grid_stack=image_plane_grid_stack, galaxies=[lens_galaxy])
source_plane_grid_stack = image_plane.trace_grid_stack_to_next_plane()

# plt.scatter(x=pix_grid.regular_grid[:,1], y=pix_grid.regular_grid[:,0])
# plt.show()

mapper = adaptive.mapper_from_grid_stack_and_border(grid_stack=source_plane_grid_stack, border=None)

inversion = inv.Inversion(image_1d=lens_data.image_1d, noise_map_1d=lens_data.noise_map_1d,
                          convolver=lens_data.convolver_mapping_matrix, mapper=mapper,
                          regularization=reg.Constant(coefficients=(1.0,)))

# Now lets plot our rectangular mapper with the image.
# mapper_plotters.plot_image_and_mapper(ccd_data=ccd_data, mapper=mapper, mask=mask, should_plot_grid=True)

# Okay, so lets think about the rectangular pixelization. Is it the optimal way to pixelize our source plane? Are there
# features in the source-plane that arn't ideal? How do you think we could do a better job?

# Well, given we're doing a whole tutorial on using a different pixelization to the rectangular grid, you've probably
# guessed that it isn't optimal. Infact, its pretty rubbish, and not a pixelization we'll actually want to model
# many lenses with!

# So what is wrong with the grid? Well, first, lets think about the source reconstruction.
# inversion_plotters.plot_reconstructed_pixelization(inversion=inversion, should_plot_centres=True)

source_galaxy = g.Galaxy(redshift=1.0, pixelization=adaptive, regularization=reg.Constant(coefficients=(1.0,)))
tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=image_plane_grid_stack)
fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)
lens_fit_plotters.plot_fit_subplot(fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True,
                                   should_plot_image_plane_pix=True)