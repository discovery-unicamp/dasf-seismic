"""Texture attributes testing parameters"""

from dasf_seismic.attributes.texture import (
    GLCMGeneric,
    GLCMContrast,
    GLCMDissimilarity,
    GLCMASM,
    GLCMMean,
    GLCMCorrelation,
    GLCMHomogeneity,
    GLCMVariance,
    GLCMStandardDeviation,
    GLCMEnergy,
    GLCMEntropy,
    LocalBinaryPattern2D,
    LocalBinaryPattern3D,
)


attributes = [
    {
        "operator_cls": GLCMGeneric,
        "operator_params": {"glcm_type": "contrast"},
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_skip": True, # V2 is tested in GLCMContrast, V1 is kept to test the base class
        "v3_skip": True,
    },
    {
        "operator_cls": GLCMContrast,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_contrast.npy",
    },
    {
        "operator_cls": GLCMDissimilarity,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_dissimilarity.npy",
    },
    {
        "operator_cls": GLCMASM,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_asm.npy",
    },
    {
        "operator_cls": GLCMMean,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_mean.npy",
    },
    {
        "operator_cls": GLCMCorrelation,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_skip": True, # Correaltion is currently not matching - V3 test deactivated
    },
    {
        "operator_cls": GLCMHomogeneity,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_homogeneity.npy",
    },
    {
        "operator_cls": GLCMVariance,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_var.npy",
    },
    {
        "operator_cls": GLCMStandardDeviation,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_std.npy",
    },
    {
        "operator_cls": GLCMEnergy,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_energy.npy",
    },
    {
        "operator_cls": GLCMEntropy,
        "v1_in_shape": (40, 40, 40),
        "v1_in_shape_chunks": (10, 10, 10),
        "v2_in_shape_chunks": (10, 50, 50),
        "v2_slice": (50, 50, 50),
        "v2_valid_precision": -6,
        "v3_operator_params": {"glb_mi": -32767 , "glb_ma": 32767}, # GLCM uses global MIN and MAX values, V3 uses only a subvolume of F3 to execute faster
        "v3_in_shape_chunks": (10, 50, 50),
        "v3_remove_border": tuple([slice(None, None), slice(3, -3), slice(3, -3)]),  # borders aren't compared
        "v3_input": "glcm_input.npy",
        "v3_expected": "glcm_entropy.npy",
    },
    {
        "operator_cls": LocalBinaryPattern2D,
        "v3_skip": True,
    },
    {
        "operator_cls": LocalBinaryPattern3D,
        "v3_skip": True,
    },
]
