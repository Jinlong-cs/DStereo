import os
from pathlib import Path
import logging
from typing import List, Tuple
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SceneFlowMatcher:
    def __init__(self, root_dir: str):
        """
        初始化SceneFlow数据集匹配器
        
        Args:
            root_dir: SceneFlow数据集根目录
        """
        self.root_dir = Path(root_dir)
        self.output_file = "DStereoDataset_train.txt"
        
    def find_matching_files(self) -> List[Tuple[str, str, str]]:
        """
        查找匹配的左右目图像和视差图
        
        Returns:
            List[Tuple[str, str, str]]: 匹配的文件路径列表 (左目图, 右目图, 视差图)
        """
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
                        str(left_image),
                        str(right_image),
                        str(disp_image)
                    ))
                else:
                    logger.warning(f"Incomplete match for {left_image}")
        
        return matches
    
    def save_pairs(self, matches: List[Tuple[str, str, str]]):
        """
        保存匹配对到文件
        
        Args:
            matches: 匹配的文件路径列表
        """
        with open(self.output_file, 'w') as f:
            for left, right, disp in matches:
                f.write(f"{left} {right} {disp}\n")
        
        logger.info(f"Total {len(matches)} pairs saved to {self.output_file}")
        
    def process(self):
        """
        处理完整的匹配流程
        """
        logger.info("Starting SceneFlow matching process...")
        matches = self.find_matching_files()
        self.save_pairs(matches)
        logger.info("Matching process completed!")

if __name__ == "__main__":
    # 使用实际的SceneFlow数据集路径
    sceneflow_root = "/mnt/data/StereoDepthDatasets/SceneFlow/"
    matcher = SceneFlowMatcher(sceneflow_root)
    matcher.process()