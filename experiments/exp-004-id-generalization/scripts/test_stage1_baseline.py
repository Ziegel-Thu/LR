import unittest

from stage1_baseline import build_run_grid, parse_weight_decays, select_shard


class Stage1BaselineArgsTest(unittest.TestCase):
    def test_parse_weight_decays_accepts_comma_separated_values(self):
        self.assertEqual(parse_weight_decays("0,1e-5,5e-4"), [0.0, 1e-5, 5e-4])

    def test_select_shard_partitions_grid_without_overlap(self):
        grid = build_run_grid([0.0, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2], 3)
        shards = [select_shard(grid, i, 4) for i in range(4)]
        flattened = [item for shard in shards for item in shard]

        self.assertEqual(len(grid), 21)
        self.assertEqual(len(flattened), 21)
        self.assertEqual(set(flattened), set(grid))
        self.assertTrue(max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1)


if __name__ == "__main__":
    unittest.main()
