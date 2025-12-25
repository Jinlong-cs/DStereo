import colorlog
import hashlib
import logging
import requests
import random
import string
import io
import time
import os
import sys
import json
import psutil
import traceback
import signal
from pathlib import Path
from time import sleep
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional, Generator, Union, BinaryIO, Set
from dataclasses import dataclass, field
# Setup logging
logger = colorlog.getLogger()

# Remove existing handlers to avoid duplicates
for handler in logger.handlers[:]: 
    logger.removeHandler(handler)
    
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
)
logger.addHandler(handler)

# Add file handler for persistent logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"clouddisk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  # Default log level
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class CloudDiskClient:
    def create_directories_layered_parallel(self, folder_list, stats=None, already_created=0, max_workers=32):
        """
        并发分层创建目录，folder_list 可以是 Path 或 (src, dest) tuple，自动兼容 generate/local 两种模式。
        支持断点续传：already_created 表示已创建的目录数量。
        stats: 断点续传状态对象，若传入则每层创建后自动同步 folders_created 并保存 checkpoint。
        """
        # 兼容两种输入
        if not folder_list:
            return
        if isinstance(folder_list[0], tuple):
            folder_list = [item[1] for item in folder_list]

        # 跳过已创建的部分
        if already_created > 0:
            folder_list = folder_list[already_created:]
        if not folder_list:
            logger.info("No new directories to create (all done by checkpoint)")
            return

        # 按深度分层
        depth_map = defaultdict(list)
        for folder in folder_list:
            depth_map[len(folder.parts)].append(folder)

        for depth in sorted(depth_map.keys()):
            to_create = depth_map[depth]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(lambda d: self.create_directory(d, retry=5, delay=0.5), to_create))
            created_count = sum(1 for r in results if r is True)
            failed_count = len(results) - created_count
            # 同步断点续传进度
            if stats is not None:
                stats.folders_created += created_count
                if hasattr(stats, 'folders_created_error'):
                    stats.folders_created_error += failed_count
                else:
                    stats.folders_created_error = failed_count
                stats.save_checkpoint()
            if failed_count > 0:
                logger.warning(f"Layer(depth={depth}): {created_count} created, {failed_count} failed")
            else:
                logger.info(f"Layer(depth={depth}): all {created_count} directories created")


    def __init__(self, base_url: str, token: str, workspace_id: int, max_workers: int, timeout: Tuple[int, int] = (5, 60)) -> None:
        self.base_url: str = base_url
        self.token: str = token
        self.workspace_id: int = workspace_id
        self.headers: Dict[str, str] = {"Authorization": self.token}
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max(100, max_workers * 2),  # 连接池大小
            pool_maxsize=max(100, max_workers * 2),      # 每个连接的最大请求数
            max_retries=5                                # 自动重试
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({"Authorization": self.token})
        self.timeout: Tuple[int, int] = timeout

    def calculate_sha256(self, data: Union[Path, bytes, BinaryIO]) -> str:
        """Calculate SHA256 hash for file path or binary data or file-like object"""
        sha256_hash = hashlib.sha256()
        
        if isinstance(data, Path):
            with data.open("rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        elif isinstance(data, bytes):
            sha256_hash.update(data)
        else:  # File-like object
            data.seek(0)
            for chunk in iter(lambda: data.read(4096), b""):
                sha256_hash.update(chunk)
            data.seek(0)
            
        return sha256_hash.hexdigest()
    def upload_file(self, file_data: Union[Path, Tuple[str, bytes]], target_path: Path, retry: int = 5) -> bool:
        """Upload a file to the cloud storage.
        
        Args:
            file_data: Either a Path object or a tuple of (filename, bytes)
            target_path: Target directory path
            retry: Number of retry attempts
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        url: str = f"{self.base_url}/file/uploadFile"
        
        # Handle different input types
        if isinstance(file_data, Path):
            file_name = file_data.name
            file_sha256 = self.calculate_sha256(file_data)
            file_content = file_data.open("rb")
        else:
            file_name, file_content_bytes = file_data
            file_content = io.BytesIO(file_content_bytes)
            file_sha256 = self.calculate_sha256(file_content)
            
        payload: Dict[str, Any] = {
            "workspaceId": self.workspace_id,
            "sha256": file_sha256,
            "dir": str(target_path),
        }
        
        # 确保显示完整的目录路径
        full_path = str(target_path / file_name) if hasattr(target_path, 'parts') else f"{target_path}/{file_name}"
        logger.debug(f"Uploading {file_name} to {target_path}...")
        logger.info(f"Uploading file to {full_path}")
        files: Dict[str, Tuple[str, Any]] = {"file": (file_name, file_content)}
        
        for attempt in range(retry):
            try:
                # 添加超时设置，避免连接挂起
                response: requests.Response = self.session.post(
                    url, headers=self.headers, data=payload, files=files,
                    timeout=self.timeout  # 使用实例的超时设置
                )
                result: Dict[str, Any] = response.json()
                if result["status"] == 0:
                    logger.info(f"Uploaded {file_name} successfully.")
                    # Close file if it's a file object
                    if isinstance(file_data, Path):
                        file_content.close()
                    return True
                logger.error(f"Error uploading {file_name}: {result['message']}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}): {e}")
            sleep(attempt * 2)  # Exponential backoff
            
        # Close file if it's a file object
        if isinstance(file_data, Path):
            file_content.close()
        return False
    def create_directory(self, target_path: Path, retry: int = 3, delay: float = 0.5) -> bool:
        """Create a directory in the cloud storage.
        
        Args:
            target_path: Target directory path
            retry: Number of retry attempts
            delay: Delay between retries in seconds
            
        Returns:
            bool: True if directory was created successfully
        """
        url: str = f"{self.base_url}/file/createDir"
        payload: Dict[str, Any]
        
        if len(target_path.parts) < 2:
            payload = {
                "workspaceId": self.workspace_id,
                "path": "",
                "name": target_path.name,
            }
        else:
            payload = {
                "workspaceId": self.workspace_id,
                "path": str(target_path.parent),
                "name": target_path.name,
            }
            
        # Try to create directory with retries
        for attempt in range(retry):
            try:
                # 添加超时设置，避免连接挂起
                response: requests.Response = self.session.post(
                    url, headers=self.headers, json=payload,
                    timeout=self.timeout  # 使用实例的超时设置
                )
                result: Dict[str, Any] = response.json()
                
                if result["status"] == 0:
                    logger.debug(f"Created directory {target_path} successfully")
                    # Add a small delay after successful creation to ensure it's registered in the system
                    sleep(delay)
                    return True
                else:
                    error_msg = result.get('message', 'Unknown error')
                    # 兼容中英文各种“已存在”表达
                    already_exists_keywords = [
                        "already exists", "已经存在", "文件夹已经存在", "目录已经存在", "folder exists", "directory exists"
                    ]
                    if any(keyword in error_msg.lower() for keyword in already_exists_keywords):
                        logger.debug(f"Directory {target_path} already exists (msg: {error_msg})")
                        return True
                    
                    logger.warning(f"Error creating directory {target_path} (attempt {attempt+1}/{retry}): {error_msg}")
            except Exception as e:
                logger.warning(f"Exception creating directory {target_path} (attempt {attempt+1}/{retry}): {e}")
                if 'response' in locals() and hasattr(response, "text"):
                    logger.warning(response.text)
                    
            # Wait before retry
            sleep(delay * (attempt + 1))  # Exponential backoff
            
        logger.error(f"Failed to create directory {target_path} after {retry} attempts")
        return False
        
    def directory_exists(self, path: Path) -> bool:
        """Check if a directory exists in the cloud storage
        
        Args:
            path: Path to check
            
        Returns:
            bool: True if directory exists
        """
        # Since there's no direct API to check directory existence,
        # try to create it and check if it already exists
        url: str = f"{self.base_url}/file/createDir"
        
        if len(path.parts) < 2:
            payload = {
                "workspaceId": self.workspace_id,
                "path": "",
                "name": path.name,
            }
        else:
            payload = {
                "workspaceId": self.workspace_id,
                "path": str(path.parent),
                "name": path.name,
            }
            
        try:
            response = self.session.post(url, headers=self.headers, json=payload)
            result = response.json()
            
            # Log the response for debugging
            logger.debug(f"Directory existence check for {path}: status={result.get('status')}, message={result.get('message', 'No message')}")
            
            # If creation was successful or directory already exists
            # 检查状态码和消息（包括中文"文件夹已经存在"和英文"already exists"）
            if (result["status"] == 0 or 
                result["status"] == 100019 or  # 文件夹已经存在的状态码
                "already exists" in result.get('message', '').lower() or
                "文件夹已经存在" in result.get('message', '')):
                logger.debug(f"Directory {path} exists or was created successfully")
                return True
            
            # Log the failure reason
            logger.warning(f"Directory {path} does not exist. API response: {result}")
            return False
        except Exception as e:
            logger.error(f"Error checking if directory {path} exists: {e}")
            if 'response' in locals() and hasattr(response, "text"):
                logger.error(f"Response text: {response.text}")
            return False
    def generate_random_data(self, size: int = 50 * 1024) -> bytes:
        """Generate random binary data with specified size.
        
        Args:
            size: Size of the data in bytes, defaults to 50KB
            
        Returns:
            bytes: Random binary data
        """
        return os.urandom(size)
        
    def generate_random_filename(self, prefix: str = "file", ext: str = "dat") -> str:
        """Generate a random filename.
        
        Args:
            prefix: Prefix for the filename
            ext: File extension
            
        Returns:
            str: Random filename
        """
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"{prefix}_{random_str}.{ext}"
    
    @dataclass
    class FileUploadTask:
        """Class for holding file upload task data"""
        filename: str
        content: bytes = field(repr=False)  # Don't include content in repr to avoid huge log entries
        target_path: Path
        size: int = field(default=0, init=False)
        task_id: str = field(default="", init=False)
        attempts: int = field(default=0, init=False)
        
        def __post_init__(self):
            self.size = len(self.content)
            # Create a unique task ID based on path and filename
            self.task_id = hashlib.md5(f"{self.target_path}/{self.filename}".encode()).hexdigest()
            
        def get_upload_tuple(self):
            """Get the tuple needed for upload_file method"""
            return ((self.filename, self.content), self.target_path)
            
    @dataclass
    class UploadStats:
        """Class for tracking upload statistics"""
        total_files_target: int = 0
        files_generated: int = 0
        folders_created: int = 0
        upload_succeeded: int = 0
        upload_failed: int = 0
        bytes_uploaded: int = 0
        start_time: float = field(default_factory=time.time)
        last_report_time: float = field(default_factory=time.time)
        checkpoint_file: Path = Path("upload_checkpoint.json")
        completed_tasks: Set[str] = field(default_factory=set)
        
        def report_progress(self, force: bool = False):
            """Report progress statistics"""
            now = time.time()
            # Report every 5 seconds or when forced
            if force or (now - self.last_report_time) >= 5:
                elapsed = now - self.start_time
                mb_uploaded = self.bytes_uploaded / (1024 * 1024)
                upload_pct = 0 if self.files_generated == 0 else (self.upload_succeeded / self.files_generated) * 100
                
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                
                logger.info(
                    f"Progress: Generated {self.files_generated}/{self.total_files_target} files, "
                    f"Created {self.folders_created} folders, "
                    f"Uploaded {self.upload_succeeded}/{self.files_generated} files ({upload_pct:.1f}%), "
                    f"Failed: {self.upload_failed}, "
                    f"Data: {mb_uploaded:.2f} MB, "
                    f"Elapsed: {elapsed:.1f}s, "
                    f"Memory: {memory_usage:.1f} MB"
                )
                self.last_report_time = now
                
        def save_checkpoint(self):
            """Save progress to checkpoint file"""
            data = {
                "files_generated": self.files_generated,
                "folders_created": self.folders_created,
                "upload_succeeded": self.upload_succeeded,
                "upload_failed": self.upload_failed,
                "bytes_uploaded": self.bytes_uploaded,
                "completed_tasks": list(self.completed_tasks),
                "timestamp": time.time()
            }
            
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f)
                
            # Atomic replace to avoid corruption if interrupted
            temp_file.replace(self.checkpoint_file)
            logger.info(f"Checkpoint saved to {self.checkpoint_file}")
            
        def load_checkpoint(self) -> bool:
            """Load progress from checkpoint file"""
            if not self.checkpoint_file.exists():
                return False
                
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    
                self.files_generated = data.get("files_generated", 0)
                self.folders_created = data.get("folders_created", 0)
                self.upload_succeeded = data.get("upload_succeeded", 0)
                self.upload_failed = data.get("upload_failed", 0)
                self.bytes_uploaded = data.get("bytes_uploaded", 0)
                self.completed_tasks = set(data.get("completed_tasks", []))
                
                timestamp = data.get("timestamp", 0)
                age = time.time() - timestamp
                logger.info(f"Loaded checkpoint from {self.checkpoint_file} (age: {age:.1f}s)")
                logger.info(f"Resuming from {self.files_generated} files generated, "
                          f"{self.upload_succeeded} files uploaded")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False
        
    def upload_generated_structure(
        self, 
        target_folder: Path = Path("/"), 
        max_depth: int = 20, 
        folders_per_level: int = 5,
        total_files_target: int = 1000000,
        file_size: int = 50 * 1024,
        max_workers: int = 0,
        resume: bool = True,
        checkpoint_file: str = "upload_checkpoint.json",
        batch_size: int = 1000,
        report_interval: int = 5,
        memory_limit_mb: int = 4096
    ) -> None:
        """Generate and upload a random folder structure with specified parameters.
        
        Args:
            target_folder: Root folder where the structure will be uploaded
            max_depth: Maximum depth of folder structure
            files_per_folder: Number of files to create in each folder
            folders_per_level: Number of subfolders to create in each folder
            total_files_target: Target total number of files to create
            file_size: Size of each file in bytes
            max_workers: Number of parallel workers for uploads (0 for auto-detection)
            resume: Whether to resume from previous checkpoint
            checkpoint_file: Path to checkpoint file
            batch_size: Number of files to process in each batch
            report_interval: Seconds between progress reports
            memory_limit_mb: Memory usage limit in MB before throttling
        """
        # Set up signal handlers for graceful shutdown
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        
        # Auto-detect max workers if not specified
        if max_workers <= 0:
            max_workers = max(4, min(16, cpu_count()))
            logger.info(f"Auto-detected {max_workers} workers based on system CPU count")
            
        # Initialize statistics and checkpointing
        stats = self.UploadStats(total_files_target=total_files_taƒrget)
        stats.checkpoint_file = Path(checkpoint_file)
        
        # Convert string paths to Path objects
        if isinstance(target_folder, str):
            target_folder = Path(target_folder)
        
        # For resuming purposes
        resumed = False
        if resume and stats.load_checkpoint():
            resumed = True
            
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, saving checkpoint and exiting...")
            stats.save_checkpoint()
            # Restore original handlers and re-raise
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
            if sig == signal.SIGINT:
                raise KeyboardInterrupt()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            logger.info(f"Starting upload to {target_folder} with parameters:")
            logger.info(f"  Max depth: {max_depth}, Folders per level: {folders_per_level}")
            logger.info(f"  Target: {total_files_target} files, File size: {file_size/1024:.1f}KB")
            logger.info(f"  Workers: {max_workers}, Total expected data: {(total_files_target * file_size)/(1024*1024*1024):.2f} GB")
            
            # Calculate folder distribution to reach target file count
            total_folders = sum(folders_per_level**d for d in range(1, max_depth+1))
            files_per_folder = max(1, int(total_files_target / total_folders))
            
            logger.info(f"Estimated total folders: {total_folders}, files per folder: {files_per_folder}")
            
            # Create task queue for generation and processing
            task_queue = []
            
            # Calculate expected total folders
            expected_total_folders = sum(folders_per_level**d for d in range(1, max_depth+1))
            
            # Check if we need to create directories
            if resumed and stats.folders_created >= expected_total_folders:
                logger.info(f"PHASE 1: SKIPPED - Already created {stats.folders_created} directories according to checkpoint")
                logger.info("Proceeding directly to file generation and upload...")
            else:
                # Step 1: First create the entire directory structure
                logger.info("PHASE 1: Creating directory structure...")
                
                # Track directories to be created
                directories_to_create = []
                
                def plan_directory_structure(path, depth):
                    """Generate the plan for directory structure without creating them yet"""
                    nonlocal directories_to_create
                    
                    if depth >= max_depth:
                        return
                    
                    # Plan subfolders
                    for i in range(folders_per_level):
                        folder_name = f"folder_{depth}_{i}"
                        new_path = path / folder_name
                        
                        # Add to list of directories to create
                        directories_to_create.append(new_path)
                        
                        # Recursively plan deeper structure
                        plan_directory_structure(new_path, depth + 1)
                
                import json, os
                # 目录缓存参数
                folder_cache_file = getattr(args, 'folder_cache', 'folder_scan_cache.json')
                use_folder_cache = getattr(args, 'use_folder_cache', False)
                clear_folder_cache = getattr(args, 'clear_folder_cache', False)
                # 清理缓存
                if clear_folder_cache and os.path.exists(folder_cache_file):
                    os.remove(folder_cache_file)
                # 尝试加载缓存
                if use_folder_cache and os.path.exists(folder_cache_file):
                    with open(folder_cache_file) as f:
                        directories_to_create = [Path(p) for p in json.load(f)]
                    logger.info(f"Loaded {len(directories_to_create)} folders from cache: {folder_cache_file}")
                else:
                    # Create the plan for directories
                    plan_directory_structure(target_folder, 0)
                    logger.info(f"Directory structure planned with {len(directories_to_create)} folders")
                    if use_folder_cache:
                        with open(folder_cache_file, "w") as f:
                            json.dump([str(p) for p in directories_to_create], f)
                        logger.info(f"Saved folder structure to cache: {folder_cache_file}")
                # Create directories breadth-first (parent directories first)
                # Sort folders by depth and path string to ensure absolute stable order
                directories_to_create.sort(key=lambda p: (len(p.parts), str(p)))
                
                # Create directories with verification

                
                # 新并发分层创建目录逻辑
                self.create_directories_layered_parallel(directories_to_create, stats=stats, already_created=stats.folders_created)
                logger.info(f"PHASE 1 COMPLETE: Created {len(directories_to_create)} directories (parallel by layer)")
            
            # Step 2: Generate and upload files
            logger.info("PHASE 2: Generating and uploading files...")
            
            # Function to generate files for a given path
            def generate_files_for_path(path, depth):
                nonlocal task_queue
                
                if depth >= max_depth or stats.files_generated >= total_files_target:
                    return
                
                # Create files in current folder
                for _ in range(files_per_folder):
                    if stats.files_generated >= total_files_target:
                        break
                        
                    # Generate filename and unique task ID
                    filename = self.generate_random_filename()
                    task_id = hashlib.md5(f"{path}/{filename}".encode()).hexdigest()
                    
                    # Skip if this task was already completed in a previous run
                    if task_id in stats.completed_tasks:
                        continue
                        
                    # 目录已在第一阶段创建，无需再次验证
                    # Generate random data
                    content = self.generate_random_data(file_size)
                    
                    # Create task and add to queue
                    task = self.FileUploadTask(filename, content, path)
                    task_queue.append(task)
                    stats.files_generated += 1
                    
                    # Process batch if queue is full to avoid excessive memory usage
                    if len(task_queue) >= batch_size:
                        process_batch()
                    
                    # Regular progress reporting
                    stats.report_progress()
                    
                    # Memory check and throttling
                    process = psutil.Process(os.getpid())
                    memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                    if memory_usage_mb > memory_limit_mb:
                        logger.warning(f"Memory usage ({memory_usage_mb:.1f}MB) exceeds limit "
                                     f"({memory_limit_mb}MB). Processing current batch...")
                        process_batch()
                        # Force garbage collection
                        import gc, time
                        t0 = time.time()
                        gc.collect()
                        logger.info(f"Manual GC took {time.time() - t0:.3f}s")
                
                # Recursively process subfolders
                for i in range(folders_per_level):
                    if stats.files_generated >= total_files_target:
                        break
                        
                    folder_name = f"folder_{depth}_{i}"
                    new_path = path / folder_name
                    
                    # Process this folder
                    generate_files_for_path(new_path, depth + 1)
            
            # Function to process a batch of upload tasks
            def process_batch():
                nonlocal task_queue
                
                if not task_queue:
                    return
                    
                logger.info(f"Processing batch of {len(task_queue)} files...")
                
                # Create upload tasks
                upload_tasks = [task.get_upload_tuple() for task in task_queue]
                task_ids = [task.task_id for task in task_queue]
                file_sizes = [task.size for task in task_queue]
                
                # Free memory by clearing the queue (tasks are now in upload_tasks)
                task_sizes = sum(len(task.content) for task in task_queue)
                task_queue = []
                
                # Upload files in parallel
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(self.upload_file, file_data, target_path) 
                                  for (file_data, target_path) in upload_tasks]
                        
                        # Process results as they complete
                        for i, future in enumerate(futures):
                            try:
                                result = future.result()
                                if result:
                                    stats.upload_succeeded += 1
                                    stats.bytes_uploaded += file_sizes[i]
                                    stats.completed_tasks.add(task_ids[i])
                                else:
                                    stats.upload_failed += 1
                            except Exception as e:
                                logger.error(f"Upload task failed with error: {e}")
                                stats.upload_failed += 1
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                
                # Save checkpoint after each batch
                stats.save_checkpoint()
                stats.report_progress(force=True)
            
            # Start the generation and upload process
            if not resumed:
                # Execute the two-phase process
                # Phase 1: Already completed above - directory structure creation
                # Phase 2: Generate and upload files
                generate_files_for_path(target_folder, 0)
            else:
                logger.info(f"Resuming from checkpoint with {stats.files_generated} files generated")
                logger.info("Continuing file generation and upload...")
                generate_files_for_path(target_folder, 0)
            
            # Process any remaining tasks
            if task_queue:
                process_batch()
                
            # Final report
            elapsed_time = time.time() - stats.start_time
            success_rate = 0 if stats.files_generated == 0 else (stats.upload_succeeded / stats.files_generated) * 100
            mb_uploaded = stats.bytes_uploaded / (1024 * 1024)
            gb_uploaded = mb_uploaded / 1024
            
            logger.info("=============== Upload Summary ===============")
            logger.info(f"Created {stats.folders_created} folders")
            logger.info(f"Generated {stats.files_generated} files")
            logger.info(f"Successfully uploaded {stats.upload_succeeded} files ({success_rate:.1f}%)")
            logger.info(f"Failed uploads: {stats.upload_failed}")
            logger.info(f"Total data uploaded: {gb_uploaded:.2f} GB")
            logger.info(f"Total time: {elapsed_time:.2f} seconds")
            logger.info(f"Average speed: {mb_uploaded/elapsed_time:.2f} MB/s")
            logger.info("==============================================")
            
        except Exception as e:
            logger.critical(f"Fatal error in upload process: {e}")
            logger.critical(traceback.format_exc())
            stats.save_checkpoint()
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
    
    def upload_folder(
        self, 
        source_folder: Path, 
        target_folder: Path, 
        max_workers: int = 0,
        resume: bool = True,
        checkpoint_file: str = "upload_local_checkpoint.json",
        batch_size: int = 100,
        memory_limit_mb: int = 4096,
        use_folder_cache: bool = False,
        clear_folder_cache: bool = False,
        folder_cache_file: str = "folder_scan_cache.json"
    ) -> None:
        """Upload existing local folder to cloud storage with advanced features.
        
        Args:
            source_folder: Local folder to upload
            target_folder: Target folder in cloud storage
            max_workers: Number of parallel workers for uploads (0 for auto-detection)
            resume: Whether to resume from previous checkpoint
            checkpoint_file: Path to checkpoint file
            batch_size: Number of files to process in each batch
            memory_limit_mb: Memory usage limit in MB before throttling
        """
        # Set up signal handlers for graceful shutdown
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigterm_handler = signal.getsignal(signal.SIGTERM)
        
        # Auto-detect max workers if not specified
        if max_workers <= 0:
            max_workers = max(4, min(16, cpu_count()))
            logger.info(f"Auto-detected {max_workers} workers based on system CPU count")
            
        # Convert string paths to Path objects
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)
        if isinstance(target_folder, str):
            target_folder = Path(target_folder)
            
        # Initialize statistics and checkpointing
        stats = self.UploadStats()
        stats.checkpoint_file = Path(checkpoint_file)
        
        # For resuming purposes
        resumed = False
        if resume and stats.load_checkpoint():
            resumed = True
            
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.warning(f"Received signal {sig}, saving checkpoint and exiting...")
            stats.save_checkpoint()
            # Restore original handlers and re-raise
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
            if sig == signal.SIGINT:
                raise KeyboardInterrupt()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        import json, os
        try:
            if not source_folder.exists():
                logger.error(f"Source folder {source_folder} does not exist")
                return
                
            logger.info(f"Starting upload from {source_folder} to {target_folder}")
            
            # First, collect folder structure and statistics
            logger.info("PHASE 1: Scanning local folder structure...")

            # 目录缓存参数处理
            if clear_folder_cache and os.path.exists(folder_cache_file):
                os.remove(folder_cache_file)
                logger.info(f"Cleared folder scan cache: {folder_cache_file}")

            all_folders = []
            file_tasks = []
            completed_tasks = stats.completed_tasks

            if use_folder_cache and os.path.exists(folder_cache_file):
                with open(folder_cache_file, "r") as f:
                    cache = json.load(f)
                    all_folders = [(Path(src), Path(dst)) for src, dst in cache.get("all_folders", [])]
                    file_tasks = [(Path(src), Path(dst), tid, sz) for src, dst, tid, sz in cache.get("file_tasks", [])]
                logger.info(f"Loaded folder/file scan cache: {folder_cache_file} ({len(all_folders)} folders, {len(file_tasks)} files)")
            else:
                # Use a queue for BFS folder scanning to avoid recursion stack limits
                folder_queue = [(source_folder, target_folder)]
                all_folders = [(source_folder, target_folder)]  # Track all folders for directory creation
                file_tasks = []  # Track all files to upload
                
                # If resuming, load completed tasks
                completed_tasks = stats.completed_tasks
                
                while folder_queue:
                    src_folder, dest_folder = folder_queue.pop(0)
                    
                    try:
                        for item in src_folder.iterdir():
                            if item.is_dir():
                                # Queue this folder for processing
                                new_dest = dest_folder / item.name
                                folder_queue.append((item, new_dest))
                                all_folders.append((item, new_dest))
                            else:
                                # Generate a task ID for this file
                                task_id = hashlib.md5(f"{dest_folder}/{item.name}".encode()).hexdigest()
                                
                                # Skip if already uploaded in a previous run
                                if task_id in completed_tasks:
                                    continue
                                    
                                # Add to file tasks
                                file_size = item.stat().st_size
                                file_tasks.append((item, dest_folder, task_id, file_size))
                    except PermissionError as e:
                        logger.warning(f"Permission error scanning {src_folder}: {e}")
                    except Exception as e:
                        logger.warning(f"Error scanning {src_folder}: {e}")
                # 保存缓存
                if use_folder_cache:
                    with open(folder_cache_file, "w") as f:
                        json.dump({
                            "all_folders": [[str(src), str(dst)] for src, dst in all_folders],
                            "file_tasks": [[str(src), str(dst), tid, sz] for src, dst, tid, sz in file_tasks]
                        }, f)
                    logger.info(f"Saved folder/file scan cache: {folder_cache_file} ({len(all_folders)} folders, {len(file_tasks)} files)")

            # Update statistics
            total_files = len(file_tasks)
            total_size = sum(size for _, _, _, size in file_tasks)
            
            stats.total_files_target = total_files
            stats.files_generated = total_files  # For local upload, all files are "generated" at start
            
            logger.info(f"Found {len(all_folders)} folders and {total_files} files")
            logger.info(f"Total data to upload: {total_size/(1024*1024):.2f} MB")
            
            # Create directories if not resuming or if not all directories created
            if not resumed or stats.folders_created < len(all_folders):
                logger.info("PHASE 2: Creating directory structure...")
                
                # Sort folders by depth and path string to ensure absolute stable order
                all_folders.sort(key=lambda p: (len(p[1].parts), str(p[1])))
                
                # 新并发分层创建目录逻辑
                self.create_directories_layered_parallel([d for _, d in all_folders], stats=stats, already_created=stats.folders_created)
                logger.info(f"Directory structure created with {len(all_folders)} folders (parallel by layer)")
            else:
                logger.info(f"PHASE 2: SKIPPED - Already created {stats.folders_created} directories according to checkpoint")
            
            # Upload files in batches
            logger.info("PHASE 3: Uploading files...")
            
            # Process files in batches to manage memory usage
            for i in range(0, len(file_tasks), batch_size):
                batch = file_tasks[i:i+batch_size]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_tasks)-1)//batch_size + 1} with {len(batch)} files")
                
                # Upload files in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    
                    for src_file, dest_folder, task_id, file_size in batch:
                        # Skip if already uploaded in a previous run
                        if task_id in stats.completed_tasks:
                            continue
                            
                        futures.append(executor.submit(self.upload_file, src_file, dest_folder))
                    
                    # Process results as they complete
                    for i, future in enumerate(futures):
                        try:
                            result = future.result()
                            src_file, dest_folder, task_id, file_size = batch[i]
                            
                            if result:
                                stats.upload_succeeded += 1
                                stats.bytes_uploaded += file_size
                                stats.completed_tasks.add(task_id)
                            else:
                                stats.upload_failed += 1
                                logger.warning(f"Failed to upload {src_file} to {dest_folder}")
                        except Exception as e:
                            stats.upload_failed += 1
                            logger.error(f"Upload task failed with error: {e}")
                
                # Report progress and save checkpoint after each batch
                stats.report_progress(force=True)
                stats.save_checkpoint()
                
                # Check memory usage and throttle if necessary
                process = psutil.Process(os.getpid())
                memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                if memory_usage_mb > memory_limit_mb:
                    logger.warning(f"Memory usage ({memory_usage_mb:.1f}MB) exceeds limit. Forcing garbage collection...")
                    import gc
                    gc.collect()
            
            # Final report
            elapsed_time = time.time() - stats.start_time
            success_rate = 0 if total_files == 0 else (stats.upload_succeeded / total_files) * 100
            mb_uploaded = stats.bytes_uploaded / (1024 * 1024)
            gb_uploaded = mb_uploaded / 1024
            
            logger.info("=============== Upload Summary ===============")
            logger.info(f"Created {stats.folders_created} folders")
            logger.info(f"Successfully uploaded {stats.upload_succeeded}/{total_files} files ({success_rate:.1f}%)")
            logger.info(f"Failed uploads: {stats.upload_failed}")
            logger.info(f"Total data uploaded: {gb_uploaded:.2f} GB")
            logger.info(f"Total time: {elapsed_time:.2f} seconds")
            logger.info(f"Average speed: {mb_uploaded/elapsed_time:.2f} MB/s")
            logger.info("==============================================")
            
        except Exception as e:
            logger.critical(f"Fatal error in upload process: {e}")
            logger.critical(traceback.format_exc())
            stats.save_checkpoint()
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint_handler)
            signal.signal(signal.SIGTERM, original_sigterm_handler)
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cloud storage upload test with random data generation or local folder upload")
    parser.add_argument("--url", default="https://cloud-test2.d-robotics.cc/api/cloudDiskApi", help="API base URL")
    parser.add_argument("--workspace", type=int, default=0, help="Workspace ID")
    parser.add_argument("--token", default="YOUR_TOKEN_HERE", help="API token")
    parser.add_argument("--target", default="/", help="Target folder path in cloud storage")
    parser.add_argument("--timeout-connect", type=int, default=5, help="Connection timeout in seconds")
    parser.add_argument("--timeout-upload", type=int, default=60, help="Upload (read) timeout in seconds")
    # 新增缓存相关参数
    parser.add_argument("--use-folder-cache", action="store_true", help="Use folder/file scan cache for local mode")
    parser.add_argument("--clear-folder-cache", action="store_true", help="Clear folder/file scan cache before scanning")
    parser.add_argument("--folder-cache", default="folder_scan_cache.json", help="Folder/file scan cache file name")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Random data generation mode
    gen_parser = subparsers.add_parser("generate", help="Generate random data and upload")
    gen_parser.add_argument("--depth", type=int, default=10, help="Maximum folder depth")
    gen_parser.add_argument("--folders-per-level", type=int, default=2, help="Number of subfolders per folder")
    gen_parser.add_argument("--total-files", type=int, default=1000000, help="Total target number of files")
    gen_parser.add_argument("--file-size", type=int, default=50, help="Size of each file in KB")
    gen_parser.add_argument("--checkpoint", default="upload_checkpoint.json", help="Checkpoint file path")
    
    # Local folder upload mode
    local_parser = subparsers.add_parser("local", help="Upload local folder structure")
    local_parser.add_argument("--source", required=True, help="Source local folder to upload")
    local_parser.add_argument("--checkpoint", default="upload_local_checkpoint.json", help="Checkpoint file path")
    local_parser.add_argument("--folder-cache", type=str, default="folder_scan_cache.json", help="Directory analysis cache file")
    local_parser.add_argument("--use-folder-cache", action="store_true", help="Whether to enable directory analysis cache")
    local_parser.add_argument("--clear-folder-cache", action="store_true", help="Whether to clear directory analysis cache")
    
    # Common arguments for both modes
    for subparser in [gen_parser, local_parser]:
        subparser.add_argument("--workers", type=int, default=0, help="Number of parallel workers (0 for auto)")
        subparser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
        subparser.add_argument("--memory-limit", type=int, default=4096, help="Memory usage limit in MB")
        subparser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    
    # Global options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.setLevel(getattr(logging, args.log_level))
    logger.info(f"Log level set to {args.log_level}")
    
    # Display system information
    logger.info(f"System information:")
    logger.info(f"  Python version: {sys.version.split()[0]}")
    logger.info(f"  CPU cores: {cpu_count()}")
    memory = psutil.virtual_memory()
    logger.info(f"  Total memory: {memory.total / (1024**3):.1f} GB")
    logger.info(f"  Available memory: {memory.available / (1024**3):.1f} GB")
    
    # Default to generate mode if not specified
    if not args.mode:
        args.mode = "generate"
        logger.info("No mode specified, defaulting to 'generate' mode")
    
    # Create client
    client = CloudDiskClient(
        args.url, args.token, args.workspace, 
        max_workers=getattr(args, 'workers', 0),
        timeout=(args.timeout_connect, args.timeout_upload)
    )
    
    try:
        if args.mode == "generate":
            logger.info("Running in GENERATE mode - creating and uploading random data")
            # Execute upload with parameters from command line
            client.upload_generated_structure(
                target_folder=args.target,
                max_depth=args.depth,
                folders_per_level=args.folders_per_level,
                total_files_target=args.total_files,
                file_size=args.file_size * 1024,  # Convert KB to bytes
                max_workers=args.workers,
                resume=not args.no_resume,
                checkpoint_file=args.checkpoint,
                batch_size=args.batch_size,
                memory_limit_mb=args.memory_limit
            )
            logger.info(f"参数汇总: workers={args.workers}, batch_size={args.batch_size}, memory_limit={args.memory_limit}MB, checkpoint={args.checkpoint}, CPU核数={cpu_count()}")
        elif args.mode == "local":
            logger.info(f"Running in LOCAL mode - uploading existing local folder")
            logger.info(f"参数汇总: workers={args.workers}, batch_size={args.batch_size}, memory_limit={args.memory_limit}MB, checkpoint={args.checkpoint}, folder_cache={getattr(args, 'folder_cache', 'folder_scan_cache.json')}, use_folder_cache={getattr(args, 'use_folder_cache', False)}, clear_folder_cache={getattr(args, 'clear_folder_cache', False)}, CPU核数={cpu_count()}")
            # Execute local folder upload
            client.upload_folder(
                source_folder=args.source,
                target_folder=args.target,
                max_workers=args.workers,
                resume=not args.no_resume,
                checkpoint_file=args.checkpoint,
                batch_size=args.batch_size,
                memory_limit_mb=args.memory_limit,
                use_folder_cache=getattr(args, 'use_folder_cache', False),
                clear_folder_cache=getattr(args, 'clear_folder_cache', False),
                folder_cache_file=getattr(args, 'folder_cache', 'folder_scan_cache.json')
            )
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)
    
    logger.info("Process completed")
    sys.exit(0)
 