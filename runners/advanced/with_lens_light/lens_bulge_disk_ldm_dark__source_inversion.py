import os

### WELCOME ###

# Welcome to the pipeline runner, which loads a strong lens dataset and analyses it using a lens modeling pipeline.

# This script uses an 'advanced' pipeline. I'll be assuming that you are familiar with 'beginner' and 'intermediate'
# pipelines, so if anything isn't clear check back to the their runners and pipelines!

# First, lets consider some aspects of beginner and intermediate pipelines that were sub-optimal:

# - They often repeated the same phases to initialize the lens model. For example, source inversion pipelines using
#   hyper-features always repeated the same 4 phases to initialize the inversion (use a magnification based pixelization
#   and then surface-brightness based one). This lead to lots of repetition and wasted processing time!

# - Tweaking a pipeline to slightly change the model it fitted often required a rerun of the entire pipeline or
#   for one to create a new pipeline that slightly changed its behaviour.

# Advanced address these problems using pipeline composition, whereby multiple different pipelines each focused on
# fitting a specific aspect of the lens model are added together. The results of the initial pipelines are then reused
# when changing the model fitted in later pipelines. For example, we may initialize a source inversion using such a
# pipeline and use this to fit a variety of different mass models with different pipelines.

### SLAM (Source, Light and Mass) ###

# Advanced pipelines are written following the SLAM method, whereby we first initialize the source fit to a lens,
# followed by the lens's light and then the lens's mass.

# If you think back to the beginner and intermediate pipelines this is exactly what they di. We'd get the inversion
# up and running first, then refine the lens's light model and finally its mass. Advanced pipelines simply use
# separate pipelines to do this, each of which has features that enable more customization of the model that is fitted.

### THIS RUNNER ###

# Using two source pipelines and a mass pipeline we will fit a power-law mass model and source using a pixelized
# inversion.

# We'll use the example pipelines:
# 'autolens_workspace/pipelines/advanced/with_lens_light/source/parametric/lens_bulge_disk_sie__source_sersic.py'.
# 'autolens_workspace/pipelines/advanced/with_lens_light/source/inversion/from_parametric/lens_bulge_disk_sie__source_inversion.py'.
# 'autolens_workspace/pipelines/advanced/with_lens_light/light/bulge_disk/lens_bulge_disk_sie__source.py'.
# 'autolens_workspace/pipelines/advanced/with_lens_light/mass/light_dark/lens_bulge_disk_mlr_nfw__source.py'.

# Check them out now for a detailed description of the analysis!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###
import autolens as al
import autolens.plot as aplt

# Specify the dataset label and name, which we use to determine the path we load the data from.
dataset_label = "imaging"
dataset_name = "lens_sersic_exp_mlr_nfw__source_sersic"
pixel_scales = 0.1

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# Using the dataset path, load the data (image, noise-map, PSF) as an imaging object from .fits files.
imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

# Next, we create the mask we'll fit this data-set with.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# Make a quick subplot to make sure the data looks as we expect.
aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)


### PIPELINE SETUP + SETTINGS ###

# Advanced pipelines still use general settings, which customize the hyper-mode features and inclusion of a shear.

pipeline_general_settings = al.PipelineGeneralSettings(
    hyper_galaxies=True,
    hyper_image_sky=False,
    hyper_background_noise=True,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
)

# We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

### SOURCE ###

from pipelines.advanced.with_lens_light.source.parametric import (
    lens_bulge_disk_sie__source_sersic,
)
from pipelines.advanced.with_lens_light.source.inversion.from_parametric import (
    lens_bulge_disk_sie__source_inversion,
)

# Advanced pipelines also use settings which specifically customize the source, lens light and mass analyses. You've
# seen the source settings before, which for this pipeline are shown below and define:

# - The Pixelization used by the inversion of this pipeline (and all pipelines that follow).
# - The Regularization scheme used by of this pipeline (and all pipelines that follow).
# - Whether the lens light and mass centres are fixed to input values (see tools/data_making/imaging/lens_light_centre_maker.py)
# - Whether the centres of the light and mass profiles are aligned.
# - Whether the lens light profile is fixed in later phases after initialization.

pipeline_source_settings = al.PipelineSourceSettings(
    lens_light_centre=(0.0, 0.0),
    lens_mass_centre=(0.0, 0.0),
    align_light_mass_centre=False,
    lens_light_bulge_only=True,
    no_shear=False,
    fix_lens_light=True,
)

pipeline_source__parametric = lens_bulge_disk_sie__source_sersic.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    pipeline_source_settings=pipeline_source_settings,
    phase_folders=["advanced", dataset_label, dataset_name],
)

pipeline_source__inversion = lens_bulge_disk_sie__source_inversion.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    pipeline_source_settings=pipeline_source_settings,
    phase_folders=["advanced", dataset_label, dataset_name],
)

### Light ###

from pipelines.advanced.with_lens_light.light.bulge_disk import (
    lens_bulge_disk_sie__source,
)

# The mass settings for this pipeline are shown below, which define:

# - Whether the centre, axis ratio and / or rotation angle of the bulge and disk are aligned.

pipeline_light_settings = al.PipelineLightSettings(
    align_bulge_disk_centre=True,
    align_bulge_disk_axis_ratio=False,
    align_bulge_disk_phi=False,
)

pipeline_light__bulge_disk = lens_bulge_disk_sie__source.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    pipeline_light_settings=pipeline_light_settings,
    phase_folders=["advanced", dataset_label, dataset_name],
)

### MASS ###

from pipelines.advanced.with_lens_light.mass.light_dark import (
    lens_light_mlr_nfw__source,
)

# The mass settings for this pipeline are shown below, which define:

# - Whether the lens light profile is fixed whilst modeling the mass.

pipeline_mass_settings = al.PipelineMassSettings(fix_lens_light=False)

pipeline_mass__power_law = lens_light_mlr_nfw__source.make_pipeline(
    pipeline_general_settings=pipeline_general_settings,
    pipeline_light_settings=pipeline_light_settings,
    pipeline_mass_settings=pipeline_mass_settings,
    phase_folders=["advanced", dataset_label, dataset_name],
)

### PIPELINE COMPOSITION AND RUN ###

# We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
# information throughout the analysis to later phases.

pipeline = (
    pipeline_source__parametric
    + pipeline_source__inversion
    + pipeline_light__bulge_disk
    + pipeline_mass__power_law
)

pipeline.run(dataset=imaging, mask=mask)
