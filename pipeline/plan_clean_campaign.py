"""Prepare explicit clean-run commands and measured resource estimates; never execute training."""
import json
import math
from pathlib import Path
from research import ROOT


def main():
    profile_path=ROOT/'experiments/training_resource_profile_batch6.json'
    profile=json.loads(profile_path.read_text())
    measured={r['experiment']:r for r in profile['results']}
    split=json.loads((ROOT/'datasets/current_research_split.json').read_text())
    version=split['version']
    runs=[]
    for experiment,conditioning,label in [(4,'class_only','morphology'),(5,'class_only','mechanism'),
                                          (6,'class_only','matched-five-channel-control'),(6,'spatial','spatial')]:
        for seed in (2026,2027,2028):
            run_id=f'clean-{version.removeprefix("research_")}-exp{experiment}-{label}-seed{seed}'
            command=['python','pipeline/train_research.py','--experiment',str(experiment),'--conditioning',conditioning,
                     '--split-version',version,'--run-id',run_id,'--seed',str(seed),
                     '--epochs','50','--batch-size','6']
            counts=split['datasets']['exp4_level_a_morphology_256' if experiment==4 else 'exp5_exp6_level_bc_mechanisms_256']
            if experiment in (5,6):
                counts=split['quality_views']['exp5_exp6_level_bc_mechanisms_256']['split_counts']
            seconds=sum(math.ceil(counts[role]/6)*measured[experiment]['timings'][role]['median_batch_seconds'] for role in ('train','validation'))
            runs.append(dict(run_id=run_id,command=command,estimated_hours=seconds*50/3600,
                             estimate_proxy=experiment==6 and conditioning=='class_only'))
    plan=dict(status='prepared_not_launched',runs=runs,estimated_total_training_hours=sum(r['estimated_hours'] for r in runs),
              profile_source=str(profile_path.relative_to(ROOT)),
              memory_profile='Measured batch-6 allocated memory below 4 GiB; see native GPU profile',
              comparison='Exp6 class_only and spatial share the same five-channel architecture, seven labels, dataset and preprocessing; the class-only control receives constant null fields.',
              historical_ladder='Exp4 and Exp5 change label spaces and are not a controlled head-to-head comparison.',
              estimate_limits=profile['estimate_limitations']+' Estimates recalculate current split row counts using the historical v4 batch profile. The class-only five-channel control uses the spatial run throughput as a planning proxy.',
              data_quality='Exp5/6 use strict_rc_geometry_v1 under v5; flagged RC annotation records are excluded without changing base splits or historical data.',
              prerequisites=['Choose explicit GPU-time budget before launching a multi-day campaign',
                             'Independent label review and generated-image detector validation remain needed for strong mechanism claims',
                             'Review conservative similarity grouping limitations before claiming dataset independence'],
              evaluation_target='200 images per class per seed; class-only/spatial paired sampling noise; frozen test set; per-seed and class-level uncertainty')
    out=ROOT/'experiments/clean_campaign_plan.json'
    out.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(dict(runs=len(runs),estimated_total_training_hours=plan['estimated_total_training_hours'],output=str(out),status=plan['status']),indent=2))


if __name__=='__main__':main()
