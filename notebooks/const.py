# Species codes that match each damage agent.
# From 10.1016/j.foreco.2025.122549
HOST_SPCODES = {
    "mtn_pb": (113,108,122,101,116,117,119,109,102,142,114,133,104,105,106),
    "fir_eng": (15,20,17,22),
    "pin_ips": (106, 133),
    "spr_bet": (93, 96, 97),
    "west_pb": (122,109,112,117,116,101),
    "west_bb": (19, 18),
    "doug_fb": (202,),
    "jeff_pb": (116, 109, 137)
}

# Cross walk from damage-causing insect to the DCA
# code used in ADS damage polygons.
HOST_DCA_CODES = {
    "mtn_pb": 11006,
    "fir_eng": 11050,
    "pin_ips": 11019,
    "spr_bet": 11009,
    "west_pb": 11002,
    "west_bb": 11015,
    "doug_fb": 11007,
    "jeff_pb": 11004
}

# How many pixels to aggregate from the 30 m TreeMap dataset. This controls
# the output resolution (and thus all the downstream datasets too).
COARSEN_FACTOR = 100

# Output projection for gridded data
PROJECTION = "EPSG:5070"

# Covariates used by the GBM
GBM_COVARIATES = {
    "hydro": ["HT", "P50", "WUE", "rdmax", "gsmax"],
    "topo": ["elev", "heat"],
    "climate": ["tmin", "vpd", "def"]
}