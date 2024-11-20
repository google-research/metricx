# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluates the predictions from a MetricX model."""

import collections
import dataclasses
import json
import os

from mt_metrics_eval import data
from mt_metrics_eval import tasks
import numpy as np
import transformers


@dataclasses.dataclass
class Arguments:
  en_de: str = dataclasses.field(metadata={"help": "The en-de input file."})
  en_es: str = dataclasses.field(metadata={"help": "The en-es input file."})
  ja_zh: str = dataclasses.field(metadata={"help": "The ja-zh input file."})

  output_file: str = dataclasses.field(
      metadata={"help": "The output file with evaluation metrics."},
  )


def _load_scores(
    input_file: str, num_segments: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
  """Loads segment and system-level scores."""
  scores = collections.defaultdict(dict)
  with open(input_file, "r") as f:
    for line in f:
      instance = json.loads(line)
      system_id = instance["system_id"]
      segment_id = instance["segment_id"]
      score = -1 * instance["prediction"]
      scores[system_id][segment_id] = score

  seg_scores = {}
  for system_id in scores:
    seg_scores[system_id] = []
    for segment_id in range(num_segments):
      seg_scores[system_id].append(scores[system_id].get(segment_id, None))

  sys_scores = {}
  for system_id in seg_scores:
    cur_scores = np.asarray(seg_scores[system_id])
    sys_scores[system_id] = np.mean(cur_scores[cur_scores != None])  # pylint: disable=singleton-comparison

  return seg_scores, sys_scores


def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()

  # Download MTME data
  data.Download()

  metric_name = "metricx-24-v2p6"
  wmt24_lps = ["en-de", "en-es", "ja-zh"]
  evs_dict = {
      ("wmt24", lp): data.EvalSet("wmt24", lp, True) for lp in wmt24_lps
  }

  segment_counts_per_lp = {}
  for lp in wmt24_lps:
    evs = evs_dict[("wmt24", lp)]
    gold_scores = evs.Scores("seg", "mqm")
    for _, scores in gold_scores.items():
      segment_counts_per_lp[lp] = len(scores)
      continue
  scores = {
      "en-de": _load_scores(args.en_de, segment_counts_per_lp["en-de"]),
      "en-es": _load_scores(args.en_es, segment_counts_per_lp["en-es"]),
      "ja-zh": _load_scores(args.ja_zh, segment_counts_per_lp["ja-zh"]),
  }

  for lp in wmt24_lps:
    evs = evs_dict[("wmt24", lp)]
    seg_scores, sys_scores = scores[lp]
    evs._scores["seg"][f"{metric_name}-{evs.std_ref}"] = seg_scores  # pylint: disable=protected-access
    evs._scores["sys"][f"{metric_name}-{evs.std_ref}"] = sys_scores  # pylint: disable=protected-access
    evs._metric_names.add(f"{metric_name}-{evs.std_ref}")  # pylint: disable=protected-access
    evs._metric_basenames.add(metric_name)  # pylint: disable=protected-access

  for evs in evs_dict.values():
    evs.SetPrimaryMetrics(evs.primary_metrics | {metric_name})

  wmt24_tasks, wts = tasks.WMT24(wmt24_lps, k=0)
  results = wmt24_tasks.Run(eval_set_dict=evs_dict)
  metrics = {"average_correlation": results.AverageCorrs(wts)[metric_name]}

  if args.output_file:
    dirname = os.path.dirname(args.output_file)
    if dirname:
      os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, "w") as out:
      out.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
  main()
