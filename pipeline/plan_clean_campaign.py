"""Prepare explicit clean-run commands and measured resource estimates; never execute training."""
import json
from pathlib import Path
from research import ROOT


def main():
    profile_path=ROOT/'experiments/training_resource_profile_batch6.json'
    profile=json.loads(profile_path.read_text())
    measured={r['experiment']:r for r in profile['results']}
    runs=[]
    for experiment,conditioning,label in [(4,'class_only','morphology'),(5,'class_only','mechanism'),
                                          (6,'class_only','matched-five-channel-control'),(6,'spatial','spatial')]:
        for seed in (2026,2027,2028):
            run_id=f'clean-v4-exp{experiment}-{label}-seed{seed}'
            command=['python','pipeline/train_research.py','--experiment',str(experiment),'--conditioning',conditioning,
                     '--split-version','research_v4','--run-id',run_id,'--seed',str(seed),
                     '--epochs','50','--batch-size','6']
            runs.append(dict(run_id=run_id,command=command,estimated_hours=measured[experiment]['estimated_50_epoch_hours'],
                             estimate_proxy=experiment==6 and conditioning=='class_only'))
    plan=dict(status='prepared_not_launched',runs=runs,estimated_total_training_hours=sum(r['estimated_hours'] for r in runs),
              profile_source=str(profile_path.relative_to(ROOT)),
              memory_profile='Measured batch-6 allocated memory below 4 GiB; see native GPU profile',
              comparison='Exp6 class_only and spatial share the same five-channel architecture, seven labels, dataset and preprocessing; the class-only control receives constant null fields.',
              historical_ladder='Exp4 and Exp5 change label spaces and are not a controlled head-to-head comparison.',
              estimate_limits=profile['estimate_limitations']+' The class-only five-channel control uses the spatial run throughput as a planning proxy.',
              prerequisites=['Choose explicit GPU-time budget before launching a multi-day campaign',
                             'Independent label review and generated-image detector validation remain needed for strong mechanism claims',
                             'Complete new-checkpoint sampling/evaluation handoff before running the full campaign'],
              evaluation_target='200 images per class per seed; class-only/spatial paired sampling noise; frozen test set; per-seed and class-level uncertainty')
    out=ROOT/'experiments/clean_campaign_plan.json'
    out.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps(dict(runs=len(runs),estimated_total_training_hours=plan['estimated_total_training_hours'],output=str(out),status=plan['status']),indent=2))


if __name__=='__main__':main()
