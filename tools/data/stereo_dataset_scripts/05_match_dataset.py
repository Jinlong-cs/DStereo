import os
from pathlib import Path
import logging
from typing import List, Tuple
from tqdm import tqdm
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseDatasetMatcher(ABC):
    def __init__(self, root_dir: str, dataset_name: str, prefix: str):
        """
        初始化数据集匹配器基类
        
        Args:
            root_dir: 数据集根目录
            dataset_name: 数据集名称
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name
        self.output_file = f"{dataset_name.lower()}_train.txt"
        self.prefix = prefix
        
    @abstractmethod
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        """
        查找匹配的左右目图像和视差图，需要子类实现具体逻辑
        
        Returns:
            List[Tuple[str, str, str]]: 匹配的文件路径列表 (左目图, 右目图, 视差图)
        """
        pass
    
    def save_pairs(self, matches: List[Tuple[str, str, str]]):
        """
        保存匹配对到文件
        
        Args:
            matches: 匹配的文件路径列表
        """
        with open(self.output_file, 'w') as f:
            for left, right, disp in matches:
                f.write(f"{self.prefix}/{left} {self.prefix}/{right} {self.prefix}/{disp}\n")
        
        logger.info(f"Total {len(matches)} pairs saved to {self.output_file}")
        
    def process(self):
        """
        处理完整的匹配流程
        """
        logger.info(f"Starting {self.dataset_name} matching process...")
        matches = self.find_matching_files()
        self.save_pairs(matches)
        logger.info("Matching process completed!")

class SceneFlowMatcher(BaseDatasetMatcher):
    def __init__(self, root_dir: str, prefix: str):
        super().__init__(root_dir, "SceneFlow", prefix)
        
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        matches = []
        # SceneFlow子数据集
        subdatasets = ['Driving', 'FlyingThings3D', 'Monkaa']
        
        for subdataset in subdatasets:
            logger.info(f"Processing {subdataset}...")
            dataset_path = self.root_dir / subdataset
            
            # 获取frames_finalpass目录下的所有左目图
            left_images = list(dataset_path.glob("frames_finalpass/**/left/*.png"))
            
            for left_image in tqdm(left_images, desc=f"Matching {subdataset}"):
                # 构建对应的右目图路径
                right_image = Path(str(left_image).replace("/left/", "/right/"))
                
                # 构建对应的视差图路径
                disp_image = Path(str(left_image)
                    .replace("frames_finalpass", "disparity")
                    .replace(".png", ".pfm"))
                
                # 验证文件是否都存在
                if right_image.exists() and disp_image.exists():
                    matches.append((
                        str(left_image.relative_to(self.root_dir)),
                        str(right_image.relative_to(self.root_dir)),
                        str(disp_image.relative_to(self.root_dir))
                    ))
                else:
                    logger.warning(f"Incomplete match for {left_image}")
        
        return matches

class FallingThingsMatcher(BaseDatasetMatcher):
    def __init__(self, root_dir: str, prefix: str):
        super().__init__(root_dir, "FallingThings", prefix)
        
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        matches = []
        # 获取所有场景目录（如kitchen_0, kitchen_1等）
        scene_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.endswith('_disp_gt')]
        
        for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
            scene_name = scene_dir.name
            disp_gt_dir = self.root_dir / f"{scene_name}_disp_gt"
            
            if not disp_gt_dir.exists():
                logger.warning(f"Disparity directory not found for {scene_name}")
                continue
                
            # 获取所有左目图像
            left_images = list(scene_dir.glob("*.left.jpg"))
            
            for left_image in left_images:
                # 构建对应的右目图路径
                right_image = left_image.with_name(left_image.name.replace('.left.', '.right.'))
                
                # 构建对应的视差图路径
                frame_id = left_image.stem.split('.')[0]  # 获取帧ID（如001999）
                disp_image = disp_gt_dir / f"{frame_id}.tiff"
                
                # 验证文件是否都存在
                if right_image.exists() and disp_image.exists():
                    # 转换为相对路径
                    matches.append((
                        str(left_image.relative_to(self.root_dir)),
                        str(right_image.relative_to(self.root_dir)),
                        str(disp_image.relative_to(self.root_dir))
                    ))
                else:
                    logger.warning(f"Incomplete match for {left_image}")
        
        return matches

class SIDODMatcher(BaseDatasetMatcher):
    def __init__(self, root_dir: str, prefix: str):
        super().__init__(root_dir, "SIDOD", prefix)
        
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        matches = []
        # 获取所有场景目录（如bakery_0, AIUE_V02_001_0等）
        scene_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.endswith('_disp_gt')]
        
        for scene_dir in tqdm(scene_dirs, desc="Processing SIDOD scenes"):
            scene_name = scene_dir.name
            disp_gt_dir = self.root_dir / f"{scene_name}_disp_gt"
            
            if not disp_gt_dir.exists():
                logger.warning(f"Disparity directory not found for {scene_name}")
                continue
                
            # 获取所有左目图像
            left_images = list(scene_dir.glob("*.left.png"))
            
            for left_image in left_images:
                # 构建对应的右目图路径
                right_image = left_image.with_name(left_image.name.replace('.left.', '.right.'))
                
                # 构建对应的视差图路径
                frame_id = left_image.stem.split('.')[0]  # 获取帧ID
                disp_image = disp_gt_dir / f"{frame_id}.tiff"
                
                # 验证文件是否都存在
                if right_image.exists() and disp_image.exists():
                    # 转换为相对路径
                    matches.append((
                        str(left_image.relative_to(self.root_dir)),
                        str(right_image.relative_to(self.root_dir)),
                        str(disp_image.relative_to(self.root_dir))
                    ))
                else:
                    logger.warning(f"Incomplete match for {left_image}")
        
        return matches

class IRSMatcher(BaseDatasetMatcher):
    def __init__(self, root_dir: str, prefix: str):
        """
        初始化IRS数据集匹配器
        
        Args:
            root_dir: IRS数据集根目录
            prefix: 数据集前缀
        """
        super().__init__(root_dir, "IRS", prefix)
        # IRS数据集的四个主要场景类别
        self.scene_categories = ['Office/Office', 'Store/Store', 'Home/Home', 'Restaurant/Restaurant']
        
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        """
        查找IRS数据集中匹配的左右目图像和视差图
        
        Returns:
            List[Tuple[str, str, str]]: 匹配的文件路径列表 (左目图, 右目图, 视差图)
        """
        matches = []
        root_path = Path(self.root_dir)
        
        # 遍历四个主要场景类别
        for category in self.scene_categories:
            category_path = root_path / category
            print(category_path)
            if not category_path.exists():
                logger.warning(f"Category directory not found: {category}")
                continue
                
            # 遍历每个场景类别下的具体场景
            scene_dirs = [d for d in category_path.iterdir() if d.is_dir()]
            
            for scene_dir in tqdm(scene_dirs, desc=f"Processing {category} scenes"):
                # 获取所有左目图像
                left_images = list(scene_dir.glob("l_*.png"))
                
                for left_image in left_images:
                    # 从左目图像文件名中提取frame_id
                    frame_id = left_image.stem.split('_')[1]  # 提取数字部分
                    
                    # 构建对应的右目图和视差图路径
                    right_image = scene_dir / f"r_{frame_id}.png"
                    disp_image = scene_dir / f"d_{frame_id}.exr"
                    
                    # 验证文件是否都存在
                    if right_image.exists() and disp_image.exists():
                        # 转换为相对路径
                        matches.append((
                            str(left_image.relative_to(self.root_dir)),
                            str(right_image.relative_to(self.root_dir)),
                            str(disp_image.relative_to(self.root_dir))
                        ))
                    else:
                        logger.warning(f"Incomplete match for {left_image}")
        
        logger.info(f"Found {len(matches)} matching pairs in IRS dataset")
        return matches

class TartanAirMatcher(BaseDatasetMatcher):
    def __init__(self, root_dir: str, prefix: str):
        """
        初始化TartanAir数据集匹配器
        
        Args:
            root_dir: TartanAir数据集根目录
            prefix: 数据集前缀
        """
        super().__init__(root_dir, "TartanAir", prefix)
        
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        """
        查找TartanAir数据集中匹配的左右目图像和视差图
        
        Returns:
            List[Tuple[str, str, str]]: 匹配的文件路径列表 (左目图, 右目图, 视差图)
        """
        matches = []
        root_path = Path(self.root_dir)
        
        # 遍历场景目录
        for scene_dir in root_path.iterdir():
            if not scene_dir.is_dir():
                continue
                
            logger.info(f"Processing scene: {scene_dir.name}")
            
            # 遍历难度级别目录
            for level_dir in scene_dir.iterdir():
                if not level_dir.is_dir():
                    continue
                    
                # 遍历目标目录
                for target_dir in level_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                        
                    # 检查必要的子目录是否存在
                    left_dir = target_dir / "image_left"
                    right_dir = target_dir / "image_right"
                    disp_dir = target_dir / "disp_gt"
                    
                    if not all([left_dir.exists(), right_dir.exists(), disp_dir.exists()]):
                        logger.warning(f"Missing required directories in {target_dir}")
                        continue
                        
                    # 检查视差图目录是否为空
                    if not any(disp_dir.iterdir()):
                        logger.warning(f"Empty disparity directory: {disp_dir}")
                        continue
                    
                    # 以视差图为基准进行匹配
                    disp_images = sorted(disp_dir.glob("*.tiff"))
                    
                    for disp_image in tqdm(disp_images, desc=f"Processing {scene_dir.name}/{level_dir.name}/{target_dir.name}"):
                        # 构建对应的左右目图像路径
                        base_name = disp_image.stem  # 不包含.tiff后缀的文件名
                        left_image = left_dir / f"{base_name}_left.png"
                        right_image = right_dir / f"{base_name}_right.png"
                        
                        # 验证文件是否都存在
                        if left_image.exists() and right_image.exists():
                            # 转换为相对路径
                            matches.append((
                                str(left_image.relative_to(self.root_dir)),
                                str(right_image.relative_to(self.root_dir)),
                                str(disp_image.relative_to(self.root_dir))
                            ))
                        else:
                            logger.warning(f"Incomplete match for disparity image {disp_image}")
                            if not left_image.exists():
                                logger.warning(f"Missing left image: {left_image}")
                            if not right_image.exists():
                                logger.warning(f"Missing right image: {right_image}")
        
        logger.info(f"Found {len(matches)} matching pairs in TartanAir dataset")
        return matches
    
def create_matcher(dataset_name: str, root_dir: str, prefix: str) -> BaseDatasetMatcher:
    """
    工厂函数：根据数据集名称创建对应的匹配器
    
    Args:
        dataset_name: 数据集名称
        root_dir: 数据集根目录
    
    Returns:
        BaseDatasetMatcher: 对应的数据集匹配器实例
    """
    matchers = {
        # "sceneflow": SceneFlowMatcher,
        # "fallingthings": FallingThingsMatcher,
        # "sidod": SIDODMatcher,
        # "irs": IRSMatcher,
        "tartanair": TartanAirMatcher,
        # 在这里添加新的数据集匹配器
    }
    
    matcher_class = matchers.get(dataset_name)
    if not matcher_class:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return matcher_class(root_dir, prefix)

if __name__ == "__main__":
    # 示例使用
    datasets = {
        # "sceneflow": {
        #     "path":"/mnt/data/StereoDepthDatasets/SceneFlow",
        #     "prefix": "SceneFlow"
        #     },
        # "fallingthings": {
        #     "path": "/mnt/data/StereoDepthDatasets/FallingThings/fat/mixed",
        #     "prefix": "FallingThings/fat/mixed"
        #     },
        # "sidod": {
        #     "path": "/mnt/data/StereoDepthDatasets/SIDOD/mixed_distractor",
        #     "prefix": "SIDOD/mixed_distractor"
        # },
        # "irs": {
        #     "path": "/mnt/data/StereoDepthDatasets/IRS",
        #     "prefix": "IRS"
        # }
        "tartanair": {
            "path": "/mnt/data/StereoDepthDatasets/TartanAir",
            "prefix": "TartanAir"
        }
    }
    
    # 处理所有数据集
    for dataset_name, root_dir in datasets.items():
        try:
            matcher = create_matcher(dataset_name, root_dir["path"],root_dir["prefix"])
            matcher.process()
        except Exception as e:
            logger.error(f"Error processing {dataset_name}: {str(e)}")