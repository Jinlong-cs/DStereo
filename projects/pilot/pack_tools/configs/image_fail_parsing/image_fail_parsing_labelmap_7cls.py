from collections import OrderedDict

dst_label = OrderedDict()
dst_label["normal"] = 0
dst_label["light_blur"] = 1
dst_label["heavy_blur"] = 2
dst_label["light_glare"] = 3
dst_label["heavy_glare"] = 4
dst_label["light_blockage"] = 5
dst_label["heavy_blockage"] = 6


src_label = OrderedDict()
src_label["normal"] = 0
src_label["light_blur"] = 1
src_label["heavy_blur"] = 2
src_label["light_glare"] = 3
src_label["heavy_glare"] = 4
src_label["light_blockage"] = 5
src_label["heavy_blockage"] = 6


color_map = OrderedDict()
color_map["normal"] = [64, 192, 0]
color_map["light_blur"] = [255, 255, 0]
color_map["heavy_blur"] = [250, 170, 30]
color_map["light_glare"] = [128, 64, 192]
color_map["heavy_glare"] = [160, 0, 0]
color_map["light_blockage"] = [0, 0, 255]
color_map["heavy_blockage"] = [0, 0, 0]
color_map["ignore"] = [70, 70, 70]
