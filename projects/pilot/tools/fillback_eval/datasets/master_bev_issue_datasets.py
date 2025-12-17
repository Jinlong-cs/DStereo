from hdflow.fillback import IssueRegressionDataset

dataset_ids = dict(
    issue=[
        # bev模型issue回归datasets样例
        IssueRegressionDataset(
            ids=5797,
            desc="master_bev",
            meta={
                "merge_key": "pilot51-算法-v1.1-自测",
            },
        ),
    ]
)
