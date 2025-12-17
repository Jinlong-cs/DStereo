from hdflow.fillback import IssueRegressionDataset

dataset_ids = dict(
    issue=[
        IssueRegressionDataset(
            ids=4541,
            flag="x03",
            desc="漏检误检",
            meta={
                "merge_key": "galaxy-算法-v18-自测",
            },
        ),
        IssueRegressionDataset(
            ids=4539,
            flag="x03",
            desc="测速测距",
            meta={
                "merge_key": "galaxy-算法-v18-自测",
            },
        ),
        # 测试数据集
        IssueRegressionDataset(
            ids=4540,
            flag="x03",
            desc="朝向压线量",
            meta={
                "merge_key": "galaxy-算法-v18-自测",
            },
        ),
    ]
)
