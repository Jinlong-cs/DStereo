from hdflow.fillback import IssueRegressionDataset

dataset_ids = dict(
    obs=[
        # 测试数据集
        IssueRegressionDataset(
            ids=3306,
            flag="x03",
            desc="test_dataset",
            meta={
                "issue_job_compared": 5588,
                "merge_key": "galaxy-J5回灌-6V感知-no-can-example",
            },
        ),
    ]
)
