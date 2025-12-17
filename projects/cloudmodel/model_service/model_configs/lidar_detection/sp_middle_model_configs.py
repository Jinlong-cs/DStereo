SpMiddleResNetFHD = """
SpMiddleResNetFHD;
subm(16, 3);
bloc(16) * 2;
conv(32, 3, 2, 1);
bloc(32) * 2;
conv(64, 3, 2, 1);
bloc(64) * 2;
conv(128, 3, 2, 011);
bloc(128) * 2;
conv(128, 311, 211);
"""

SpMiddleResNetFHD_lite1122 = """
SpMiddleResNetFHD_lite1122;
subm(16, 3);
bloc(16);
conv(32, 3, 2, 1);
bloc(32);
conv(64, 3, 2, 1);
bloc(64) * 2;
conv(128, 3, 2, 011);
bloc(128) * 2;
conv(128, 311, 211);
"""

SpMiddleResNetFHD_lite1112 = """
SpMiddleResNetFHD_lite1112;
subm(16, 3);
bloc(16);
conv(32, 3, 2, 1);
bloc(32);
conv(64, 3, 2, 1);
bloc(64);
conv(128, 3, 2, 011);
bloc(128) * 2;
conv(128, 311, 211);
"""

SpMiddleResNetFHD_lite1111 = """
SpMiddleResNetFHD_lite1111;
subm(16, 3);
bloc(16);
conv(32, 3, 2, 1);
bloc(32);
conv(64, 3, 2, 1);
bloc(64);
conv(128, 3, 2, 011);
bloc(128);
conv(128, 311, 211);
"""

SpMiddleFHDLite = """
SpMiddleFHDLite;
conv(16, 3, 1, 1);
conv(32, 3, 2, 1);
conv(64, 3, 2, 1);
conv(128, 3, 2, 011);
conv(128, 311, 211);
"""

SpMiddleFPRCNN = """
SpMiddleFPRCNN;
conv(8, 3, 2, 011);
conv(16, 233, 1, 011);
conv(32, 3, 211, 011);
conv(32, 233, 122, 011);
conv(64, 233, 2, 1);
conv(64, 233, 1, 011);
"""

SpMiddleFPRCNN_bloc = """
SpMiddleFPRCNN_bloc;
subm(8, 3);
bloc(8) * 2;
conv(16, 233, 2, 011);
bloc(16) * 2;
conv(32, 3, 211, 011);
bloc(32) * 2;
conv(32, 233, 122, 011);
bloc(32) * 2;
conv(64, 233, 2, 1);
bloc(64) * 2;
conv(64, 233, 1, 011);
bloc(64) * 2;
"""

SpMiddleFHDLarge = """
SpMiddleFHDLarge;
subm(16, 3);
subm(16, 3);
conv(32, 3, 2, 1);
subm(32, 3) * 2;
conv(64, 3, 2, 1);
subm(64, 3) * 3;
conv(128, 3, 2, 011);
subm(128, 3) * 3;
conv(128, 311, 211);
"""


SpMiddleResNetD4 = """
SpMiddleResNetD4;
subm(32, 3);
bloc(32) * 2;
conv(64, 3, 2, 1);
bloc(64) * 2;
conv(64, 3, 2, 011);
bloc(64) * 2;
conv(64, 3, 2, 011);
"""

SpMiddleResNetD4_lite122 = """
SpMiddleResNetD4_lite122;
subm(32, 3);
bloc(32);
conv(64, 3, 2, 1);
bloc(64) * 2;
conv(64, 3, 2, 011);
bloc(64) * 2;
conv(64, 3, 2, 011);
"""

SpMiddleResNetD4_lite112 = """
SpMiddleResNetD4_lite112;
subm(32, 3);
bloc(32);
conv(64, 3, 2, 1);
bloc(64);
conv(64, 3, 2, 011);
bloc(64) * 2;
conv(64, 3, 2, 011);
"""

SpMiddleResNetD4_clite = """
SpMiddleResNetD4_clite;
subm(16, 3);
bloc(16) * 2;
conv(32, 3, 2, 1);
bloc(32) * 2;
conv(64, 3, 2, 011);
bloc(64) * 2;
conv(64, 3, 2, 011);
"""

# 39 not trained
SpMiddleResNetD4_rsbn_clite111 = """
SpMiddleResNetD4_rsbn_clite111;
subm(16, 3);
rsbn(16);
conv(32, 3, 2, 1);
rsbn(32);
conv(64, 3, 2, 011);
rsbn(64);
conv(64, 3, 2, 011);
"""

#
SpMiddleResNetD4_rsbn_ex2_clite112 = """
SpMiddleResNetD4_rsbn_ex2_clite112;
subm(16, 3);
rsbn(16);
conv(32, 3, 2, 1);
rsbn(32);
conv(64, 3, 2, 011);
rsbn(64) * 2;
conv(64, 3, 2, 011);
"""

# self.expansion = 2 if planes > 64 else 4
SpMiddleMobileNetV2 = """
SpMiddleMobileNetV2;
rsbn(16);
conv(24, 3, 2, 1);
bloc(24);
conv(32, 3, 2, 1);
rsbn(32);
conv(64, 3, 2, 011);
rsbn(64);
rsbn(128);
conv(128, 311, 211);
"""

# 57
SpMiddleResNetFHD_rsbn_lite1111 = """
SpMiddleResNetFHD_rsbn_lite1111;
subm(16, 3);
rsbn(16);
conv(32, 3, 2, 1);
rsbn(32);
conv(64, 3, 2, 1);
rsbn(64);
conv(128, 3, 2, 011);
rsbn(128);
conv(128, 311, 211);
"""

# 54
SpMiddleResNetFHD_pool = """
SpMiddleResNetFHD_pool;
subm(16, 3);
bloc(16) * 2;
conv(32, 3, 2, 1);
bloc(32) * 2;
conv(64, 3, 2, 1);
bloc(64) * 2;
conv(128, 3, 2, 011);
bloc(128) * 2;
pool(0, 211);
"""

test = """
SpMiddleMobileNetV2;
rsbn(16);
conv(24, 3, 2, 1);
rsbn(24);
conv(32, 3, 2, 1);
rsbn(32);
conv(64, 3, 2, 011);
rsbn(64);
conv(128, 3, 1, 1);
rsbn(128);
conv(128, 311, 211);
"""
