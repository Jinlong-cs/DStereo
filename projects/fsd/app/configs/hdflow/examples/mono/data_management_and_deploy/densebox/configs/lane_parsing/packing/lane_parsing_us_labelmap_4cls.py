from collections import OrderedDict

dst_label = OrderedDict()  # 16.2 cls
dst_label["dashed"] = 1
dst_label["solid"] = 1
dst_label["mixed"] = 1
dst_label["wide_dashed"] = 1
dst_label["wide_solid"] = 1
dst_label["deceleration_lane"] = 1
dst_label["tidal_lane"] = 1
dst_label["botts_dots"] = 1
dst_label["Road_teeth"] = 2
dst_label["double_line"] = 3

# modify
dst_label["background"] = 255

dst_label["road"] = 0
dst_label["sidewalk"] = 0
dst_label["building"] = 0
dst_label["pothole"] = 0
dst_label["fence"] = 0
dst_label["pole"] = 0
dst_label["traffic_light"] = 0
dst_label["Traffic_Sign1"] = 0
dst_label["traffic_sign"] = 0
dst_label["vegetation"] = 0
dst_label["terrain"] = 0
dst_label["sky"] = 0
dst_label["person"] = 0
dst_label["rider"] = 0
dst_label["car"] = 0
dst_label["truck"] = 0
dst_label["bus"] = 0
dst_label["train"] = 0
dst_label["motorcycle"] = 0
dst_label["bicycle"] = 0
# ----------------
dst_label["tricycle"] = 0
dst_label["lane_marking"] = 0

dst_label["Guide_Post"] = 0
dst_label["Crosswalk_Line"] = 0
dst_label["Traffic_Arrow"] = 0
dst_label["Sign_Line"] = 0
dst_label["Guide_Line"] = 0
dst_label["Traffic_Cone"] = 0
dst_label["Bollard"] = 0
# ------------------
dst_label["Stop_Line"] = 0
dst_label["Slow_Down_Triangle"] = 0
dst_label["Speed_Sign"] = 0

dst_label["Diamond"] = 0
dst_label["BicycleSign"] = 0
dst_label["SpeedBumps"] = 0
# apa sign cls
dst_label["parking_line"] = 0
dst_label["Parking_space"] = 0
dst_label["parking_rod"] = 0
dst_label["parking_lock"] = 0
dst_label["column"] = 0
dst_label["no_forward_marker"] = 0
dst_label["traversable_obstruction"] = 0
dst_label["untraversable_obstruction"] = 0
dst_label["mask"] = 255
dst_label["other"] = 255

src_label = OrderedDict()  # 38 cls
src_label["road"] = 0
src_label["pothole"] = 0
src_label["sidewalk"] = 1
src_label["vegetation"] = 2
src_label["terrain"] = 3
src_label["pole"] = 4
src_label["traffic_sign"] = 5
src_label["Traffic_Sign1"] = 5
src_label["traffic_light"] = 6
src_label["Sign_Line"] = 7
src_label["lane_marking"] = 8
src_label["person"] = 9
src_label["rider"] = 10
src_label["bicycle"] = 11
src_label["motorcycle"] = 12
src_label["tricycle"] = 13
src_label["car"] = 14
src_label["truck"] = 15
src_label["bus"] = 16
src_label["train"] = 17
src_label["building"] = 18
src_label["fence"] = 19
src_label["sky"] = 20
src_label["Traffic_Cone"] = 21
src_label["Bollard"] = 22
src_label["Guide_Post"] = 23
src_label["Crosswalk_Line"] = 24
src_label["Traffic_Arrow"] = 25
src_label["Guide_Line"] = 26
src_label["Stop_Line"] = 27
src_label["Slow_Down_Triangle"] = 28
src_label["Speed_Sign"] = 29
src_label["Diamond"] = 30
src_label["BicycleSign"] = 31
src_label["SpeedBumps"] = 32
# add no_forward_marker
src_label["no_forward_marker"] = 18
# apa sign cls
src_label["parking_line"] = 8
src_label["Parking_space"] = 0
src_label["parking_rod"] = 18
src_label["parking_lock"] = 18
src_label["column"] = 18
# add genObj
src_label["traversable_obstruction"] = 0
src_label["untraversable_obstruction"] = 18
src_label["Road_teeth"] = 33
src_label["dashed"] = 34
src_label["solid"] = 35
src_label["mixed"] = 36
src_label["wide_dashed"] = 37
src_label["wide_solid"] = 38
src_label["deceleration_lane"] = 39
src_label["tidal_lane"] = 40
src_label["double_line"] = 41
src_label["botts_dots"] = 42

# modify
src_label["background"] = 255

src_label["mask"] = 255
src_label["other"] = 255

color_map = OrderedDict()  # 4 cls
color_map["road"] = [0, 0, 0]
color_map["dashed"] = [0, 0, 255]
color_map["Road_teeth"] = [0, 255, 0]
color_map["double_line"] = [255, 255, 0]
# lane parsing task 'other' class default is [255, 0, 0]
