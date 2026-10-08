#!/usr/bin/env python3
"""Build final career evidence from the restored combined-skill panel.
Does not rescan the source CSV or overwrite the source-restoration audit.
"""
from pathlib import Path
from html import unescape
import hashlib
import json
import platform
import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT, FIG, GEN = ROOT/'outputs/step4', ROOT/'figures/step4', ROOT/'_generated'
for folder in (OUT, FIG, GEN):
    folder.mkdir(parents=True, exist_ok=True)

# Execute the same three numeric model blocks preserved by the restoration.
# Plotly examples later in the page are not needed for the numeric model.
blocks = re.findall(r'```(?:\{python\}|python)\s*\n(.*?)\n```', (ROOT/'ml_methods.qmd').read_text(), re.S)
assert len(blocks) == 6, 'Expected the restored ML Methods page with six code examples.'
model_context = {}
for block in blocks[:3]:
    exec(compile(block, 'ml_methods.qmd', 'exec'), model_context)
panel = model_context['market_panel']
bridge = pd.read_csv(ROOT/'data/processed/step3/career_market_skills.csv')
matrix = model_context['feature_matrix']
comparison = model_context['model_comparison'].copy()
clustered = model_context['clustered_panel']
assert len(panel) == panel.JOB_ID.nunique() == 10
assert matrix.shape == (10, 37), 'Review model and report if the inputs change.'
assert panel.NAICS_2022_6.eq(513210).all()
assert pd.to_datetime(panel.POSTED_DATE).between('2024-05-01', '2024-09-30').all()
assert matrix.sum(axis=1).gt(0).all()
saved_matrix = pd.read_csv(ROOT/'outputs/step3/ml_feature_matrix.csv').set_index('JOB_ID')
pd.testing.assert_frame_equal(matrix, saved_matrix, check_names=False, check_dtype=False)
saved = pd.read_csv(ROOT/'outputs/step3/ml_cluster_assignments_with_ids.csv')
match = clustered[['JOB_ID','CLUSTER_NAME']].merge(saved[['JOB_ID','CLUSTER_NAME']], on='JOB_ID', validate='one_to_one')
assert len(match) == 10 and match.CLUSTER_NAME_x.eq(match.CLUSTER_NAME_y).all()
assert comparison['Silhouette Score'].tolist() == [0.291, 0.327, 0.311]
eligible = comparison.loc[comparison['Smallest Cluster'].ge(2)]
assert int(eligible.sort_values('Silhouette Score', ascending=False).iloc[0]['Candidate Clusters']) == 2
comparison.columns = ['Clusters','Silhouette','Smallest group','Singleton groups']
comparison['Selected'] = comparison.Clusters.eq(2).map({True:'Yes',False:'No'})
matrix.to_csv(OUT/'combined_skill_matrix.csv')


def markdown_table(frame):
    lines = ['| '+' | '.join(str(x) for x in frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(str(x).replace('|','/').replace('\n',' ') for x in row)+' |')
    return '\n'.join(lines)+'\n'


def save_table(frame, name):
    frame.to_csv(OUT/(name+'.csv'), index=False)
    (GEN/(name+'.qmd')).write_text(markdown_table(frame), encoding='utf-8')


save_table(comparison, 'model_comparison')
assignments = clustered[['JOB_ID','TITLE_RAW','ROLE_SEGMENT','SCOPE_TIER','COMPANY_NAME',
                         'SALARY_MIDPOINT_ANNUAL','MIN_YEARS_EXPERIENCE','CLUSTER_NAME']].copy()
assignments.rename(columns={'CLUSTER_NAME':'Segment'}, inplace=True)
assignments.to_csv(OUT/'segment_assignments.csv', index=False)
segment_order = ['Analytics and Data','AI/ML Architecture']
segment_rows = []
for name in segment_order:
    part = assignments.loc[assignments.Segment.eq(name)]
    segment_rows.append({'Segment':name,'Postings':len(part),
                         'Salary coverage':f'{part.SALARY_MIDPOINT_ANNUAL.count()}/{len(part)}',
                         'Median pay':f'${part.SALARY_MIDPOINT_ANNUAL.median():,.0f}',
                         'Median min years':int(part.MIN_YEARS_EXPERIENCE.median())})
save_table(pd.DataFrame(segment_rows), 'segment_summary')


def clean_description(value):
    text = unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9+#./-]+', ' ', text)
    return ' '.join(text.lower().split())


corpus = pd.read_csv(ROOT/'data/processed/step3/career_market_descriptions.csv')
assert len(corpus) == corpus.JOB_ID.nunique() == 10
corpus = corpus.set_index('JOB_ID').reindex(panel.JOB_ID)
assert corpus.BODY.fillna('').str.strip().ne('').all()
patterns = json.loads((ROOT/'scripts/skill_patterns.json').read_text())
clean_body = corpus.BODY.map(clean_description)
text_features = pd.DataFrame({skill:clean_body.str.contains(pattern,regex=True).astype(int) for skill,pattern in patterns.items()})
old_mentions = pd.read_csv(ROOT/'outputs/step3/nlp_description_skill_mentions.csv')
actual_pairs = set()
for skill in text_features.columns:
    for job_id in text_features.index[text_features[skill].eq(1)]:
        actual_pairs.add((job_id,skill))
assert actual_pairs == set(zip(old_mentions.JOB_ID,old_mentions.Skill))
text_features.to_csv(OUT/'narrow_text_skill_matrix.csv')

bridge['MODEL_SKILL'] = bridge.SKILL.replace(model_context['skill_aliases'])
frequencies = bridge.groupby('MODEL_SKILL').JOB_ID.nunique()
source_counts = bridge.loc[bridge.SKILL_ORIGIN.isin(['Both','Structured field'])].groupby('MODEL_SKILL').JOB_ID.nunique()
profile = {'Python':3,'SQL':3,'Machine Learning':2,'MLOps':1,'AWS':2,'Generative AI':2,'Data Modeling':3,'Data Visualization':3}
gap_rows, source_rows = [], []
for skill, rating in profile.items():
    count = int(frequencies.loc[skill])
    priority = 5 if count>=5 else 4 if count>=4 else 3 if count>=3 else 2 if count>=2 else 1
    gap_rows.append({'Skill':skill,'Postings':f'{count}/10','Priority':priority,'Our rating':rating,'Gap':priority-rating})
    source_rows.append({'Skill':skill,'Source labels':int(source_counts.get(skill,0)),
                        'Narrow text matches':int(text_features[skill].sum()),'Combined evidence':count})
gaps = pd.DataFrame(gap_rows).sort_values(['Gap','Skill'],ascending=[False,True])
save_table(gaps,'skill_gaps')
(GEN/'skill_gaps_report.qmd').write_text(markdown_table(gaps),encoding='utf-8')
source_comparison = pd.DataFrame(source_rows)
save_table(source_comparison,'skill_source_comparison')
assert source_comparison.set_index('Skill').loc['MLOps'].tolist() == [3,3,5]
assert gaps.set_index('Skill').loc['MLOps','Gap'] == 4

summary = matrix.groupby(np.array(clustered.CLUSTER_NAME)).sum().reindex(segment_order)
display_skills = ['SQL','Python','Data Visualization','Data Modeling','Machine Learning','MLOps','AWS','Generative AI']
skill_table = pd.DataFrame({'Skill':display_skills,
                           'Analytics and Data':[f'{int(summary.at[segment_order[0],s])}/4' for s in display_skills],
                           'AI/ML Architecture':[f'{int(summary.at[segment_order[1],s])}/6' for s in display_skills]})
save_table(skill_table,'segment_skills')
NAVY,TEAL,BLUE = '#17324D','#147D83','#386EA5'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':NAVY,
                     'text.color':NAVY,'savefig.facecolor':'white'})
fig,ax = plt.subplots(figsize=(9,3.2))
heat = summary[display_skills].div([4,6],axis=0).to_numpy()
ax.imshow(heat,cmap='Blues',vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(len(display_skills)),[s.replace(' ','\n') for s in display_skills],fontsize=9)
ax.set_yticks([0,1],['Analytics and Data (n = 4)','AI/ML Architecture (n = 6)'],fontsize=10)
for row in range(2):
    for col in range(len(display_skills)):
        count = int(summary.loc[segment_order[row],display_skills[col]])
        ax.text(col,row,f'{count}/{[4,6][row]}',ha='center',va='center',
                color='white' if heat[row,col]>.55 else NAVY,fontweight='bold')
ax.set_title('Combined skill evidence in the two segments',loc='left',pad=18)
fig.tight_layout()
fig.savefig(FIG/'segment_skills.png',dpi=200,bbox_inches='tight')
plt.close(fig)

salary = panel.loc[panel.SALARY_MIDPOINT_ANNUAL.notna()]
assert len(salary)==7 and salary.SALARY_MIDPOINT_ANNUAL.median()==173200
assert salary.SCOPE_TIER.eq('Adjacent').all()
fig,axes = plt.subplots(1,2,figsize=(9,3.4),gridspec_kw={'width_ratios':[1,1.1]})
states = panel.STATE.value_counts().sort_values()
axes[0].barh(states.index,states.values,color=TEAL)
axes[0].set_title('Where the ten postings appeared',loc='left',fontsize=11)
axes[0].set_xlabel('Postings')
axes[0].set_xticks(range(6))
sorted_pay = np.sort(salary.SALARY_MIDPOINT_ANNUAL.to_numpy())/1000
axes[1].plot(sorted_pay,range(1,8),'o',color=BLUE,markersize=8)
axes[1].axvline(np.median(sorted_pay),color=TEAL,linestyle='--')
axes[1].set_yticks(range(1,8),[f'Pay record {i}' for i in range(1,8)])
axes[1].set_title('Seven reported salary midpoints',loc='left',fontsize=11)
axes[1].set_xlabel('Annual midpoint in USD thousands')
axes[1].set_xlim(100,210)
fig.tight_layout(w_pad=2)
fig.savefig(FIG/'market_snapshot.png',dpi=200,bbox_inches='tight')
plt.close(fig)
fig,ax = plt.subplots(figsize=(8,3.4))
ordered = gaps.sort_values(['Gap','Skill'])
ax.barh(ordered.Skill,ordered.Gap,color=TEAL)
ax.set_xticks(range(5))
ax.set_xlim(0,4.7)
ax.set_xlabel('Market priority minus self rating')
ax.set_title('MLOps and machine learning lead the learning plan',loc='left',pad=12)
for i,value in enumerate(ordered.Gap):
    ax.text(value+.08,i,str(value),va='center')
fig.tight_layout()
fig.savefig(FIG/'skill_gaps.png',dpi=200,bbox_inches='tight')
plt.close(fig)

manifest = {'model_basis':'Restored combined source labels and description keywords',
            'postings':len(panel),'retained_skills':matrix.columns.tolist(),'selected_clusters':2,
            'selected_silhouette':0.291,'candidate_silhouettes':comparison.Silhouette.tolist(),
            'exact_agreement_with_restored_matrix':True,'exact_agreement_with_restored_membership':True,
            'narrow_text_pairs_reproduced':True,'full_source_rescanned_by_this_script':False,
            'source_restoration_audit_modified':False,'python':platform.python_version(),
            'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__}
for label,name in [('panel','career_market_panel.csv'),('skill_bridge','career_market_skills.csv')]:
    manifest[label+'_sha256'] = hashlib.sha256((ROOT/'data/processed/step3'/name).read_bytes()).hexdigest()
(OUT/'validation_summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FINAL ANALYTICS CHECKS PASSED')
print('Restored combined-skill model: 10 postings; 37 skills; 6/4 membership.')
print('Corrected silhouettes: 0.291 / 0.327 / 0.311; selected clusters: 2.')
print('MLOps: 3 source-label postings; 3 narrow-text matches; 5 combined-evidence postings.')
print('Source-restoration audit preserved. Outputs: outputs/step4')
