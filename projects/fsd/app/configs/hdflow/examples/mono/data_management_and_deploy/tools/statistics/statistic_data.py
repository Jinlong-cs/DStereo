import argparse
import os.path

from hdflow.data.anno_data_manage.data_stats import DataStatsEngine
from hdflow.data.anno_data_manage.data_stats.stats_render import (
    FileStatsRender,
    ImageStatsRender,
    TerminalStatsRender,
)
from mono.data_management_and_deploy.tools.statistics.utils import (
    iters_from_source,
)

from hat.utils import Config


def get_parser():
    parser = argparse.ArgumentParser(
        description="argument for collect fee excel"
    )
    parser.add_argument("--statistic-config", type=str)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--statistic-result-dir", type=str, default="./statistic_results"
    )

    args, _ = parser.parse_known_args()

    return args


if __name__ == "__main__":
    args = get_parser()

    statistic_configs = Config.fromfile(args.statistic_config)
    (
        valid_items,
        image_key_iters,
        meta_generators,
        dmp_client,
    ) = iters_from_source(statistic_configs.statistic_items)
    for image_key_iter, meta_generator, source_item in zip(
        image_key_iters, meta_generators, valid_items
    ):
        output_dir = os.path.join(
            args.statistic_result_dir, source_item.item_name
        )
        engine = DataStatsEngine(
            image_key_iter,
            meta_generator,
            renders=[
                FileStatsRender(
                    output_dir=output_dir,
                    output_file_name="statistic_results.json",
                    dmp_client=dmp_client,
                ),
                TerminalStatsRender(dmp_client=dmp_client),
                ImageStatsRender(output_dir=output_dir, dmp_client=dmp_client),
            ],
            num_worker=8,
        )
        results = engine.run()
