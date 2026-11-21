"""Aggregate local annotation geometry audit and conservative qualification."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research import ROOT


def main():
    folder=ROOT/'datasets/source_provenance'
    inventory=json.loads((folder/'rc10519_local_inventory.json').read_text())
    quality=json.loads((ROOT/'datasets/exp5_exp6_level_bc_mechanisms_256/research_v5/strict_rc_geometry_v1/audit.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    axes[0].bar(['No flagged geometry issue','Flagged geometry issue'],[inventory['local_image_files']-inventory['images_with_issues'],inventory['images_with_issues']],color=['#277a78','#ca663b'])
    axes[0].set(ylabel='Local RC source images',title='Annotation geometry screen')
    axes[0].tick_params(axis='x',labelsize=8)
    splits=['train','validation','test']
    axes[1].bar(splits,[quality['qualified_split_counts'][s] for s in splits],color='#277a78',label='Retained')
    axes[1].bar(splits,[quality['excluded_by_split'][s] for s in splits],bottom=[quality['qualified_split_counts'][s] for s in splits],color='#ca663b',label='Excluded from qualified view')
    axes[1].set(ylabel='Experiment 5/6 records',title='Base v5 split and qualified subset')
    axes[1].legend(fontsize=8)
    classes=list(quality['class_counts']['train'])
    axes[2].barh([c.replace('_',' ') for c in classes],[quality['class_counts']['train'][c] for c in classes],color='#277a78')
    axes[2].set(xlabel='Retained training records',title='Seven proxy classes retained')
    fig.suptitle('RC annotation audit and conservative Experiment 5/6 data qualification',fontsize=14)
    fig.supxlabel('Geometry flags are not proof of incorrect labels. Passing this check does not validate crack presence or structural mechanisms.',fontsize=9)
    for suffix in ('png','svg'):
        fig.savefig(folder/f'rc10519_quality.{suffix}',dpi=180)
    plt.close(fig)
    print('Saved aggregate source-quality plots; no raw dataset images included')


if __name__=='__main__':main()
