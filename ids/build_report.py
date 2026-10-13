"""Render a report from measured results; never substitutes invented metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
DATA={key:json.loads((ROOT/'datasets'/key/'results/dataset_report.json').read_text()) for key in ['unsw_nb15','cicids2017']}
NAVY='#142c45';TEAL='#079b96';BLUE='#2467bd';RED='#b44e56'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titleweight':'bold','savefig.facecolor':'white'})
for key,d in DATA.items():
 m=d['models'][d['selected_model']]
 fig,ax=plt.subplots(figsize=(7.2,2.45));x=np.arange(3)
 ax.bar(x-.17,[r['validation']['macro_f1'] for r in d['models'].values()],.34,color=BLUE,label='Validation')
 ax.bar(x+.17,[r['test']['macro_f1'] for r in d['models'].values()],.34,color=TEAL,label='Held-out test')
 ax.set_xticks(x,list(d['models']));ax.set_ylim(0,1.07);ax.set_ylabel('Macro-F1');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2,frameon=False)
 fig.tight_layout();fig.savefig(FIG/f'{key}_baselines.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(3.6,2.7));labs=m['test']['confusion_labels'];cm=np.array(m['test']['confusion_matrix'])
 ax.imshow(cm,cmap='Blues');ax.set_xticks(range(2),labs);ax.set_yticks(range(2),labs);ax.set_xlabel('Predicted');ax.set_ylabel('True')
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else NAVY,fontsize=10)
 ax.set_title(d['audit']['dataset']);fig.tight_layout();fig.savefig(FIG/f'{key}_confusion.png',dpi=220);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
for ax,(key,d) in zip(axs,DATA.items()):
 for kind,color in zip(d['models'],[BLUE,'#7785a0',TEAL]):
  curve=pd.read_csv(ROOT/'datasets'/key/'results'/f'{kind}_test_review_curve.csv')
  ax.plot(curve.review_fraction*100,curve.auto_missed_attack_rate*100,label=kind,color=color,linewidth=1.5)
  r=d['models'][kind];p=r['frozen_test_policy'];ax.scatter(p['review_fraction']*100,p['auto_missed_attack_rate']*100,color=color,s=22,zorder=3)
 ax.set_title(d['audit']['dataset']);ax.set_xlabel('Review fraction (%)');ax.set_ylabel('Automatically missed attacks (%)');ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
fig.tight_layout();fig.savefig(FIG/'review_curves.png',dpi=220);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for ax,(key,d) in zip(axs,DATA.items()):
 r=d['reliability'];ax.plot([0,1],[0,1],ls='--',color='#99a5b5');ax.plot([x['mean_confidence'] for x in r],[x['accuracy'] for x in r],marker='o',color=TEAL)
 ax.set_xlim(.45,1.01);ax.set_ylim(0,1.04);ax.set_xlabel('Mean confidence');ax.set_ylabel('Observed accuracy');ax.set_title(d['audit']['dataset']);ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(FIG/'reliability.png',dpi=220);plt.close(fig)

font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
pdfmetrics.registerFont(TTFont('DejaVu',font));pdfmetrics.registerFont(TTFont('DejaVu-Bold',bold))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCustom',fontName='DejaVu-Bold',fontSize=28,leading=34,textColor=colors.HexColor(NAVY),spaceAfter=18))
styles.add(ParagraphStyle(name='SubCustom',fontName='DejaVu',fontSize=13,leading=19,textColor=colors.HexColor(TEAL),spaceAfter=16))
styles.add(ParagraphStyle(name='HCustom',fontName='DejaVu-Bold',fontSize=17,leading=23,textColor=colors.HexColor(NAVY),spaceAfter=12))
styles.add(ParagraphStyle(name='SmallHead',fontName='DejaVu-Bold',fontSize=10.5,leading=15,textColor=colors.HexColor(NAVY),spaceBefore=13,spaceAfter=7))
styles.add(ParagraphStyle(name='BodyCustom',fontName='DejaVu',fontSize=9.4,leading=14,textColor=colors.HexColor(NAVY),spaceAfter=9))
styles.add(ParagraphStyle(name='TinyCustom',fontName='DejaVu',fontSize=7.7,leading=11,textColor=colors.HexColor('#50637a'),spaceAfter=6))
styles.add(ParagraphStyle(name='CellCustom',fontName='DejaVu',fontSize=7.6,leading=10))
styles.add(ParagraphStyle(name='HeaderCustom',fontName='DejaVu-Bold',fontSize=7.7,leading=10,textColor=colors.white))
P=lambda text,style='BodyCustom':Paragraph(escape(str(text)),styles[style])
F=lambda x:f'{x:.4f}' if x is not None else 'N/A'
PC=lambda x:f'{100*x:.2f}%' if x is not None else 'N/A'
N=lambda x:f'{int(x):,}'
story=[]
def heading(text):story.append(P(text,'HCustom'))
def para(text,small=False):story.append(P(text,'TinyCustom' if small else 'BodyCustom'))
def table(rows,widths):
 data=[[P(c,'HeaderCustom' if i==0 else 'CellCustom') for c in row] for i,row in enumerate(rows)]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f0f5fa'),colors.white]),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor(NAVY)),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#dfe7ef'))]))
 story.extend([t,Spacer(1,10)])
def image(name,width=507,height=None):
 from PIL import Image as PILImage
 w,h=PILImage.open(FIG/name).size
 story.extend([Image(str(FIG/name),width=width,height=height or width*h/w),Spacer(1,8)])
def newpage():story.append(PageBreak())
def baseline_table(d):
 rows=[['Model','Validation\nmacro-F1','Test\nmacro-F1','Attack\nrecall','FPR','Attack AP']]
 for name,r in d['models'].items():rows.append([name+(' *' if name==d['selected_model'] else ''),F(r['validation']['macro_f1']),F(r['test']['macro_f1']),PC(r['test']['per_class']['Attack']['recall']),PC(r['test']['false_positive_rate']),F(r['test']['attack_pr_auc_average_precision'])])
 table(rows,[100,81,81,85,73,87]);para('* Selected by validation macro-F1. Test outcomes never determine selection.',True)

a=DATA['unsw_nb15'];b=DATA['cicids2017'];am=a['models'][a['selected_model']];bm=b['models'][b['selected_model']]
story.append(Spacer(1,26));story.append(P('SHIFTGUARD','SubCustom'));story.append(P('Two-dataset\nevaluation report','TitleCustom'))
para('Explainable and Uncertainty-Aware Intrusion Detection under Network Distribution Shift')
story.append(Spacer(1,14));para('Prepared for Dr. Tabrej Khan | Department of Computer Science | Faculty of Engineering and Informatics | Ala-Too International University, Bishkek, Kyrgyzstan')
para('10 October 2026 (Bishkek) | Actual uploaded CSV data | One development seed (42)',True)
story.append(Spacer(1,12));heading('What the measured results show')
para(f'UNSW-NB15: the validation-selected XGBoost model achieves test macro-F1 {F(am["test"]["macro_f1"])} and attack recall {PC(am["test"]["per_class"]["Attack"]["recall"])}. Its frozen review policy refers {PC(am["frozen_test_policy"]["review_fraction"])} of test flows, exceeding the 15% source budget.')
para(f'CICIDS2017: under the Friday holdout, selected XGBoost falls from validation macro-F1 {F(bm["validation"]["macro_f1"])} to test macro-F1 {F(bm["test"]["macro_f1"])}. Attack recall is only {PC(bm["test"]["per_class"]["Attack"]["recall"])}. It automatically misses {PC(bm["frozen_test_policy"]["auto_missed_attack_rate"])} of true attacks after the uncertainty gate.')
para('These findings demonstrate a failure mode of source-trained confidence-based referral. They do not support operational reliability or successful zero-day detection. Low false-positive rates on CIC coexist with extremely poor attack recall.')
table([['Dataset','Fit rows','Calibration','Validation','Test rows'],*[ [d['audit']['dataset'],*[N(d['partition_audit'][k]['rows']) for k in ['train','calibration','validation','test']]] for d in DATA.values()]],[130,94,94,94,95])
para('Two independent feature schemas were fitted. No UNSW-to-CIC transfer was run. The dashboard displays a 5,000-row uniform test sample for each dataset; this report uses full cleaned held-out-test evaluations.',True)
newpage();heading('1. Dataset inventory and audit')
para('archive (4).zip contains four headerless UNSW raw CSVs, a feature dictionary, event metadata, and two curated partitions. The raw files are inventoried separately and are not pooled with the overlapping curated subsets.')
table([['UNSW file','Rows','Role'],*[[f['file'],N(f['rows']),'Raw inventory' if 'raw archive' in f['role'] else f['role']] for f in a['audit']['files']]],[255,85,167])
para(a['audit']['filename_caution'])
para('archive (3).zip contains eight CIC flow CSVs. Their capture-day filenames are available; timestamps, endpoint addresses and session IDs are absent from these machine-learning CSVs.')
table([['CIC file','Raw rows','Valid rows'],*[[f['file'],N(f['raw_rows']),N(f['kept_rows'])] for f in b['audit']['files']]],[329,89,89])
para(f'CIC total: {N(b["audit"]["raw_rows"])} raw rows; {N(b["audit"]["clean_rows"])} remain after measurement/label cleaning, before leakage and conflict exclusions. Raw archive member counts are reported as observed, without forcing published counts.',True)
newpage();heading('2. UNSW-NB15 measured results')
para('Supplied filename direction preserved: 82,332 development-source rows and 175,341 test-source rows. The cleaned development source is split by indivisible feature signatures into fit/calibration/validation partitions.')
baseline_table(a);image('unsw_nb15_baselines.png')
table([['Selected XGBoost policy','Measured value'],['Validation-selected threshold',F(am['selected_threshold'])],['Validation review fraction',PC(am['validation_review_fraction'])],['Full-test review fraction',PC(am['frozen_test_policy']['review_fraction'])],['Automatically missed attacks',N(am['frozen_test_policy']['auto_missed_attacks'])+' / '+N(am['test']['per_class']['Attack']['support'])],['Auto-missed attack rate',PC(am['frozen_test_policy']['auto_missed_attack_rate'])]],[260,247])
para('Random Forest has a higher test macro-F1 than XGBoost in this run, but the application retains the validation-selected XGBoost. Choosing the best test performer after observing these results would contaminate model selection.')
para(f'Audit exclusions: {N(a["audit"]["development_rows_matching_test_removed"])} development rows match test signatures; {N(a["audit"]["development_binary_conflict_rows_removed"])} development rows and {N(a["audit"]["test_binary_conflict_rows_removed"])} test rows participate in binary-label-conflicting signatures. Exclusion categories can overlap; the retained total is reported independently.',True)
newpage();heading('3. CICIDS2017 Friday shift results')
para('Development uses a uniform, seed-42 reservoir capped at 200,000 valid Monday-Thursday rows. After duplicate/conflict auditing, 186,885 sampled rows remain. Friday rows retained after measurement cleaning and detected label-conflict exclusions are held out. Destination Port and the duplicate header-length column are excluded from the 76 predictors.')
baseline_table(b);image('cicids2017_baselines.png')
table([['Selected XGBoost policy','Measured value'],['Validation-selected threshold',F(bm['selected_threshold'])],['Full-test review fraction',PC(bm['frozen_test_policy']['review_fraction'])],['Automatically missed attacks',N(bm['frozen_test_policy']['auto_missed_attacks'])+' / '+N(bm['test']['per_class']['Attack']['support'])],['Auto-missed attack rate',PC(bm['frozen_test_policy']['auto_missed_attack_rate'])],['Accepted prediction error',PC(bm['frozen_test_policy']['accepted_error'])]],[260,247])
para('Friday contains Bot, DDoS and PortScan categories absent from the Monday-Thursday source categories. This is a binary unseen-attack-type/day-shift stress test, not proof of real-world zero-day detection. The near-perfect source validation score did not transfer.')
para('The low review fraction is not a success: most attacks are confidently classified as Benign. Additional OOD/referral strategies and independent target-labelled evaluation are needed before practical use. The held-out Friday data have not been used to tune or replace the selected model.',True)
newpage();heading('4. Binary detection by attack category')
para('The models predict Attack or Benign. The following category names are ground-truth strata used for evaluation, not predicted incident categories. Correct binary fraction is attack recall for malicious rows and specificity for benign rows.')
for d in DATA.values():
 story.append(P(d['audit']['dataset'],'SmallHead'));m=d['models'][d['selected_model']]
 rows=[['Ground-truth category','Test rows','Correct binary','Review','Auto misses']]
 rows += [[r['category'],N(r['rows']),PC(r['correct_binary_fraction']),PC(r['review_fraction']),N(r['automatic_missed_attacks'])] for r in m['attack_category_binary_detection']]
 table(rows,[170,83,96,78,80])
para('UNSW Fuzzers are particularly weak despite strong aggregate attack recall. In the CIC Friday holdout, Bot detection recall is zero and every Bot flow is automatically missed. Aggregate accuracy alone would conceal these failures.',True)
newpage();heading('5. Uncertainty and review workload')
image('review_curves.png')
para('Each dot marks the frozen threshold selected on source validation with a review-budget ceiling of 15%. Test curves are diagnostic; their better-looking points must not be used to retune the reported threshold.')
para('Automatic missed-attack rate counts true attacks classified Benign outside the review queue, divided by all true attacks. Referred flows are unresolved. No perfect analyst or successful recovery is assumed.')
table([['Selected model','Review count','Attention count','Attention fraction'],*[ [d['audit']['dataset'],N(d['models'][d['selected_model']]['frozen_test_policy']['review_count']),N(d['models'][d['selected_model']]['frozen_test_policy']['attention_count']),PC(d['models'][d['selected_model']]['frozen_test_policy']['attention_fraction'])] for d in DATA.values()]],[165,100,120,122])
para('Attention combines predicted-attack alerts and uncertainty referrals, counting their overlap once. It measures flow-level demand, not unique incidents or analyst minutes. Incident aggregation, reviewer time and backlog remain pilot measurements.')
image('reliability.png')
para(f'Selected-model 10-bin ECE: UNSW {F(am["test"]["ece_10_bins"])}; CIC {F(bm["test"]["ece_10_bins"])}. Confidence calibration on source data does not guarantee target calibration. Reliability bins are empirical summaries, not confidence intervals.',True)
newpage();heading('6. Input shift and explanation evidence')
para('Input monitoring compares the fit-partition reference with the full held-out test. Numeric PSI uses training-quantile bins; categorical total variation uses the training vocabulary. Missing-value fractions are reported separately. Shift scores do not establish incident causation or confirm model failure by themselves.')
for d in DATA.values():
 story.append(P(d['audit']['dataset']+' - five largest feature shift scores','SmallHead'))
 rows=[['Feature','Metric','Shift score','Test missing']]
 rows += [[r['feature'],r['metric'],F(r['shift_score']),PC(r['missing_fraction'])] for r in d['drift'][:5]]
 table(rows,[226,90,95,96])
para('The explanation method replaces one feature with its fit-partition median/mode, then measures the change in the predicted-class probability. It is neither SHAP nor an additive decomposition. Correlated-feature replacements can be unrealistic.')
# Use a recorded, independently labelled CIC attack that was predicted Benign.
pred=pd.read_csv(ROOT/'datasets/cicids2017/results/dashboard_predictions.csv').fillna({'explanation_json':''})
candidates=pred[(pred.label==1)&(pred.prediction=='Benign')&pred.explanation_json.ne('')]
if len(candidates):
 row=candidates.iloc[0];ex=json.loads(row.explanation_json)
 story.append(P('Recorded confidently wrong CIC example','SmallHead'))
 para(f'Flow: {row.flow_id}; ground truth: {row.attack_cat}; prediction: {row.prediction}; confidence: {PC(row.confidence)}. The explanation describes why the model favours its output, not whether that output is correct.',True)
 table([['Feature','Observed','Training reference','Delta p'],*[[r['feature'],str(r['observed'])[:18],str(r['reference'])[:18],f"{r['probability_change']:+.5f}"] for r in ex[:4]]],[195,115,115,82])
else:
 para('The dashboard includes 25 recorded feature sensitivity explanations per dataset. Other rows can be explained locally using the fitted model.',True)
newpage();heading('7. Reproducibility, limits and next experiments')
para('The updated application contains six trained baseline models, an added Dataset reports screen, per-dataset audit/evaluation JSONs, baseline comparison CSVs, review curves, reliability summaries, real 5,000-row held-out sample CSVs and their prediction exports. Synthetic examples remain explicitly marked.')
para('Raw source archives and full prepared partitions are not duplicated in the application ZIP. To reproduce, place the supplied archives at known local paths and run the dataset project command with --unsw-zip and --cic-zip, or extract their CSVs into the documented raw folders. The command recreates audits, partitions, fitted models and reports.')
para('CIC cleaning: trim names; verify the duplicated Fwd Header Length column; exclude Destination Port; convert infinities and -1 initial-window sentinels to missing; drop other negative measurements and missing labels; fit medians/scales only on fit rows. Garbled web-attack label names are normalized for display. No existing CSV columns are treated as a validated UNSW/CIC common schema.')
para(f'CIC leakage audit: {N(b["audit"]["development_pool_rows_matching_test"])} valid development-pool rows match Friday signatures; {N(b["audit"]["sample_rows_matching_test_removed"])} sampled rows are removed for that reason. Binary conflicts affect {N(b["audit"]["development_pool_binary_conflict_rows"])} full development-pool rows, {N(b["audit"]["sample_binary_conflict_rows_removed"])} sampled rows and {N(b["audit"]["test_binary_conflict_rows_removed"])} test rows. These reason counts overlap; retained totals are authoritative.')
para('Limitations: one seed; no group-bootstrap confidence intervals; development subsampling for CIC; conservative feature-signature conflict exclusions can alter the evaluated population; no endpoint/session split in curated CSVs; no cross-dataset transfer; no authorized campus pilot; no live capture or automatic blocking. Timing is one predict_proba batch, excluding preprocessing/explanations, not an end-to-end p95 service measurement.')
para('Next experiments: compare raw versus calibrated referral, matched-workload random referral, entropy and independently validated OOD gates; preserve untouched target evaluation windows; test compatible PCAP-derived features from both datasets; run repeated seeds and group/time-block intervals; then conduct a scoped observation-only campus study with independent review and sampling of automatically accepted benign flows.')
para('Operational conclusion: the CIC-selected model is unsuitable for dependable attack detection in the tested Friday setting. The implemented uncertainty gate does not correct the failure. The result is valuable as a transparent research stress test, not a deployment success claim.')
para('Source provenance: filenames and member SHA-256 hashes appear in each dataset_report.json; processed-partition hashes and software versions accompany the evaluation. requirements-lock.txt records tested top-level versions. Full reports contain calibration ablations, confusion matrices, per-class scores, drift and timing.',True)
para('Primary dataset access and attribution: https://research.unsw.edu.au/projects/unsw-nb15-dataset and https://www.unb.ca/cic/datasets/ids-2017.html. Dataset facts in this report were measured from the supplied archives.',True)

def footer(canvas,doc):
 canvas.saveState();w,h=A4
 canvas.setStrokeColor(colors.HexColor('#dae5ee'));canvas.line(44,40,w-44,40)
 canvas.setFont('DejaVu',7);canvas.setFillColor(colors.HexColor('#63758b'));canvas.drawString(44,28,'ShiftGuard | Actual dataset evaluation | Research prototype');canvas.drawRightString(w-44,28,str(doc.page));canvas.restoreState()
output=OUT/'ShiftGuard_Dataset_Evaluation_Report.pdf'
SimpleDocTemplate(str(output),pagesize=A4,rightMargin=44,leftMargin=44,topMargin=42,bottomMargin=54,title='ShiftGuard - Two-dataset evaluation report',author='Prepared for Dr. Tabrej Khan').build(story,onFirstPage=footer,onLaterPages=footer)
print(output)
