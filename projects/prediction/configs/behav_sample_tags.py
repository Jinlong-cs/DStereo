# Copyright (c) Horizon Robotics. All rights reserved.
from enum import Enum


class BehavTag(Enum):
    # 行为标签
    Behavior_NonDefine = (0, -1)
    Behavior_KeepLane = (0, 0)  # 直行
    Behavior_LeftChange = (0, 1)  # 左变道
    Behavior_RightChange = (0, 2)  # 右变道
    # 障碍物类型
    Type_NonDefine = (1, -1)
    Type_Vehicle = (1, 0)  # 机动车
    Type_Cyclist = (1, 1)  # 骑车人
    # 障碍物大小
    Size_NonDefine = (2, -1)
    Size_Large = (2, 0)  # 大车
    Size_Small = (2, 1)  # 小车
    # 纵向速度
    LonVelo_NonDefine = (3, -1)
    LonVelo_Fast = (3, 0)  # 纵向速度快
    LonVelo_Slow = (3, 1)  # 纵向速度慢
    # 横向速度
    LatVelo_NonDefine = (4, -1)
    LatVelo_Fast = (4, 0)  # 横向速度快
    LatVelo_Slow = (4, 1)  # 横向速度快
    # 与车道线关系
    RelationToLaneLine_NonDefine = (5, -1)
    RelationToLaneLine_CloseToLaneLine = (5, 1)  # TODO 贴车道线
    RelationToLaneLine_OverLaneLine = (5, 2)  # 压车道线
    # 与车道中心线关系
    RelationToCenterLine_NonDefine = (6, -1)
    RelationToCenterLine_AlongCenterLine = (6, 0)  # TODO 沿中心线
    RelationToCenterLine_Weaving = (6, 1)  # 画龙
    # 自车行为
    EgoBehavior_NonDefine = (7, -1)
    EgoBehavior_KeepLane = (7, 0)  # TODO 自车直行
    EgoBehavior_LaneChange = (7, 1)  # TODO 自车换道
    # 切入切出
    Cut_NonDefine = (8, -1)
    Cut_In = (8, 0)  # Cut-in
    Cut_Out = (8, 1)  # Cut-out
    # 与自车距离
    Distance_NonDefine = (9, -1)
    Distance_Near = (9, 0)  # 近距离
    Distance_Far = (9, 1)  # 远距离
    # 与自车相对位置
    Position_NonDefine = (10, -1)
    Position_FrontNear = (10, 0)  # 前方近距离
    Position_FrontFar = (10, 1)  # 前方远距离
    Position_Back = (10, 2)  # 后方
    # 车道类型
    LaneType_NonDefine = (11, -1)
    LaneType_Straight = (11, 0)  # TODO 直道
    LaneType_Curve = (11, 1)  # TODO 弯道
    # 道路场景类型
    RoadType_NonDefine = (12, -1)
    RoadType_HWP = (12, 0)  # TODO 高速
    RoadType_UP = (12, 1)  # TODO 城区
