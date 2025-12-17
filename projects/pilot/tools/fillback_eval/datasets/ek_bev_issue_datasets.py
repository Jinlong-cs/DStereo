from hdflow.fillback import IssueRegressionDataset

dataset_ids = dict(
    issue=[
        # bev模型issue回归datasets样例
        IssueRegressionDataset(
            ids=6125,
            desc="ek_bev",
            meta={
                "merge_key": "pilot51-算法-v1.1-自测",
                "issue_job_compared": 11661,
            },
        ),
    ]
)
