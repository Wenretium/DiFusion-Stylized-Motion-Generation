# from .materials import colored_material_diffuse_BSDF as colored_material
from .materials import colored_material_relection_BSDF as colored_material

sat_factor = 1.1


JOINTS_MATS = {
    "Grey": ("Greys", 0.4, 0.7),  # Blue
    # "Default": ("Blues", 0.7, 1.0),  # Blue
    "Default": ("Greys", 0.5, 0.8),
    # "ArmsAboveHead": ("Reds", 0.6, 0.9), # Red
    "ArmsAboveHead": ("Blues", 0.7, 1.0),  # Blue
    "ArmsFolded": ("Blues", 0.7, 1.0), # Blue
    "Star": ("Wistia", 0.2, 1.0), # Yellow
    "LeftHop": ("Greens", 0.7, 1.0), # Green
    "LeanLeft": ("Greens", 0.7, 1.0), # Green
    "LeanRight": ("Greens", 0.7, 1.0), # Green
    "Teapot": ("Purples", 0.7, 0.95), # Purple
    # "Teapot": ("Blues", 0.7, 1.0),  # Blue
    "Neutral": ("Blues", 0.7, 1.0),  # Blue
    # "Neutral": ("Greys", 0.7, 0.95), # Black
    "Old": ("Blues", 0.7, 1.0),  # Blue
    "Rocket": ("Oranges", 0.6, 0.9), # Orange
    "Superman": ("Reds", 0.6, 0.9), # Red
    "SpinClock": ("Purples", 0.7, 0.95), # Purple
    "Swimming": ("Blues", 0.7, 1.0),  # Blue
    "Tiptoe": ("Purples", 0.75, 0.9), # Purple
    "WideLegs": ("Greens", 0.7, 1.0), # Green
    "Akimbo": ("Oranges", 0.6, 0.9), # Orange
    "ArmsBehindBack": ("Blues", 0.7, 1.0),  # Blue
    "BentForward": ("Wistia", 0.2, 1.0), # Yellow
    "Chicken": ("Wistia", 0.2, 1.0), # Yellow
    "DuckFoot": ("Oranges", 0.6, 0.9), # Orange
    "FlickLegs": ("Greens", 0.7, 1.0), # Green
    "HandsBetweenLegs": ("Purples", 0.75, 0.9), # Purple
    "HighKnees": ("Reds", 0.6, 0.9), # Red
    "OnPhoneLeft": ("Purples", 0.75, 0.9), # Purple
    "OnPhoneRight": ("Oranges", 0.6, 0.9), # Orange
    "Balance": ("Purples", 0.75, 0.9), # Purple
    "BigSteps": ("Reds", 0.6, 0.9), # Red
    "OnHeels": ("Oranges", 0.6, 0.9), # Orange
    "RaisedRightArm": ("Reds", 0.6, 0.9), # Red
    "ArmsBySide": ("Reds", 0.6, 0.9), # Red
    # "OnToesBentForward": ("Blues", 0.7, 1.0),  # Blue
    "OnToesBentForward": ("Reds", 0.6, 0.9), # Red
    "WalkingStickLeft": ("Oranges", 0.6, 0.9), # Orange
    "ArmsAboveHead_FW_1000_1140.npy": ("Reds", 0.6, 0.9), # Red
    "Neutral_FW_1000_1140.npy": ("Oranges", 0.6, 0.9), # Orange
    "Star_FW_1000_1140.npy": ("Wistia", 0.2, 1.0), # Yellow
    "LeftHop_FW_1000_1140.npy": ("Greens", 0.7, 1.0), # Green
    "ArmsFolded_FW_1000_1140.npy": ("Blues", 0.7, 1.0), # Blue
    "Teapot_FW_1000_1140.npy": ("Purples", 0.75, 0.9), # Purple
    "RaisedRightArm_FW_1280_1420.npy": ("Reds", 0.6, 0.9), # Red
    "OnHeels_FW_1140_1280.npy": ("Oranges", 0.6, 0.9), # Orange
    # "ArmsAboveHead_FW_1280_1420.npy": ("Purples", 0.75, 0.9), # Purple
    "mld.npy": ("Reds", 0.6, 0.9), # Red
    "momask.npy": ("Oranges", 0.6, 0.9), # Orange
    "puzzle.npy": ("Wistia", 0.2, 1.0), # Yellow
    "genmostyle.npy": ("Greens", 0.7, 1.0), # Green
    "smoodi.npy": ("Blues", 0.7, 1.0),  # Blue
    "ours.npy": ("Purples", 0.75, 0.9), # Purple
    "tmr.npy": ("Reds", 0.6, 0.9), # Red
    "hutmt.npy":  ("Blues", 0.7, 1.0),  # Blue
    "t2m.npy": ("Purples", 0.75, 0.9), # Purple
}

JOINTS_MATS.update({
    "trajectory": ("Purples", 0.9),
})

JOINTS_MATS.update({
    "ArmsFolded-1.0-FlickLegs": JOINTS_MATS["ArmsFolded"],
    "ArmsFolded-0.0-FlickLegs": JOINTS_MATS["FlickLegs"],
})
for key in [f"ArmsFolded-{alpha}-FlickLegs" for alpha in [0.75, 0.5, 0.25]]:
    mat_inter = [JOINTS_MATS["Grey"]] +\
                [JOINTS_MATS["ArmsFolded"]]*2 +\
                [JOINTS_MATS["FlickLegs"]]*2
    JOINTS_MATS.update({key: mat_inter})

    
JOINTS_MATS.update({
    "ArmsFolded-1.0-LeanRight": JOINTS_MATS["ArmsFolded"],
    "ArmsFolded-0.0-LeanRight": JOINTS_MATS["LeanRight"],
})
for key in [f"ArmsFolded-{alpha}-LeanRight" for alpha in [0.75, 0.5, 0.25]]:
    mat_inter = [JOINTS_MATS["LeanRight"]] +\
                [JOINTS_MATS["ArmsFolded"]]*2 +\
                [JOINTS_MATS["LeanRight"]]*2
    JOINTS_MATS.update({key: mat_inter})
    
JOINTS_MATS.update({
    "OnPhoneLeft-1.0-OnToesBentForward": JOINTS_MATS["OnPhoneLeft"],
    "OnPhoneLeft-0.0-OnToesBentForward": JOINTS_MATS["OnToesBentForward"],
})
for key in [f"OnPhoneLeft-{alpha}-OnToesBentForward" for alpha in [0.75, 0.5, 0.25]]:
    mat_inter = [JOINTS_MATS["OnToesBentForward"]] +\
                [JOINTS_MATS["Grey"]] +\
                [JOINTS_MATS["OnPhoneLeft"]] +\
                [JOINTS_MATS["OnToesBentForward"]]*2
    JOINTS_MATS.update({key: mat_inter})    
    
JOINTS_MATS.update({
    "Star-1.0-DuckFoot": JOINTS_MATS["Star"],
    "Star-0.0-DuckFoot": JOINTS_MATS["DuckFoot"],
})
for key in [f"Star-{alpha}-DuckFoot" for alpha in [0.75, 0.5, 0.25]]:
    mat_inter = [JOINTS_MATS["Grey"]] +\
                [JOINTS_MATS["Star"]]*2 +\
                [JOINTS_MATS["DuckFoot"]]*2
    JOINTS_MATS.update({key: mat_inter})
