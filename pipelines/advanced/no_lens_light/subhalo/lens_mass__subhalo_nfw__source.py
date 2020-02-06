import autofit as af
import autolens as al

# In this pipeline, we'll perform a subhalo analysis which determines the attempts to detect subhalos by putting
# subhalos at fixed intevals on a 2D (y,x) grid.

# The mass model and source are initialized using an already run 'source' and 'mass' pipeline.

# The pipeline is as follows:

# Phase 1:

# Perform the subhalo detection analysis.

# Lens Mass: Previous mass pipeline model.
# Source Light: Previous source pipeilne model.
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
# Prior Passing: Lens mass (instance -> previous pipeline), source light (model -> previous pipeline).
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 2:

# Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.

# Lens Mass: Previous mass pipeline model.
# Source Light: Previous source pipeilne model.
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
# Prior Passing: Lens mass & source light (model -> previous pipeline), subhalo mass (model -> phase 2).
# Notes: None


def make_pipeline(
    pipeline_general_settings,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    parallel=False,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    # A source tag distinguishes if the previous pipeline models used a parametric or inversion model for the source.

    source_tag = al.pipeline_settings.source_tag_from_source(
        source=af.last.instance.galaxies.source
    )

    pipeline_name = (
        "pipeline_subhalo__lens_power_law__subhalo_nfw__source_" + source_tag
    )

    # This pipeline's name is tagged according to whether:

    # 1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag)

    ### Phase 1 ###

    # In phase 1, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model and source parameters are held fixed to the best-fit values of the previous pipeline.
    # 2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=parallel)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(
        redshift=redshift_lens, mass=al.mp.SphericalTruncatedNFWMassToConcentration
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)

    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    # Setup the source model, which uses a variable parametric profile or fixed inversion model depending on the
    # previous pipeline.

    phase1 = GridPhase(
        phase_name="phase_1__subhalo_search__source_" + source_tag,
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last.instance.galaxies.lens,
            subhalo=subhalo,
            source=af.last.instance.galaxies.source,
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
        number_of_steps=5,
    )

    phase1.optimizer.const_efficiency_mode = False
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.5

    phase2 = al.PhaseImaging(
        phase_name="phase_2__subhalo_refine",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last[-1].model.galaxies.lens,
            subhalo=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.result.model.galaxies.subhalo.mass,
            ),
            source=af.last[-1].instance.galaxies.source,
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 80
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)