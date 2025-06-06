# TN_Group_7



## Installation

The installation section is structured into three subsections: Requirements and Dependencies, Download Poject Repository, and Download OpenNeuro Dataset.

### Requirements and Dependencies

The experimental setup used Matlab (version 24.2) with Statistics and Machine Learning Toolbox (version 24.2 ) Add-on and depends on the external libraries SPM12 (Statistical Parameter Mapping, version 7771) and rDCM toolbox (v1.5.0). The user needs a terminal (command line interface) and GitHub (installed on your system).

### Download Project Repository

Run the following commands in your console. The command will download the project code and the external dependencies. 

```
git clone https://gitlab.ethz.ch/tn_projects_fs2025/Project_7.git
cd Project_7
git submodule update --init --recursive
```

### Download Data Set from OpenNeuro

The experiments depends on data sets from OpenNeuro. The description of the data sets are shown below. The project repository provides bash scripts in data/ to download the relevant data sets from OpenNeuro. Run the following command in the project directory. Alternatively, go to Download section in the linked websites.

```
cd data
source download_depression_dataset.sh
source download_healthy_dataset.sh
source gunzip_dataset.sh
```

#### **Data Set Description**

**ds002748 (v.1.0.1):** Two sessions of resting state with closed eyes for patients with depression in treatment course (NFB, CBT or No treatment groups)

This dataset was used for the resting state network analysis for patients with mild depression. There are two session of RS with 2-month interval, 3 groups of patients: no treatment, CBT, or fmri-NFB treatment. A session consists of 100 dynamic scans with TR = 2.5 s and 25 slices.

https://openneuro.org/datasets/ds003007/versions/1.0.1

**ds002748 (v.1.0.5):** Resting state with closed eyes for patients with depression and healthy participants

This dataset was used for the resting state network analysis. There are 51 subjects with mild depression and 21 healthy controls. A session consist of 100 dynamic scans with TR = 2.5 s and 25 slices.

https://openneuro.org/datasets/ds002748/versions/1.0.5

### ⚠️ IMPORTANT
Only download the subjects `sub-51` up to `sub-72` to receive the healthy subjects.

## Guideline

Open Matlab and go to the project repository. Next, run following command. This will run the pipeline in right order.

```
for preprocessing:
run('src/fmri_prepro_depression.m')
run('src/fmri_prepro_healthy.m')

for statistical analysis:
run('src/rDCM_healthy_vs_depressed_analysis.m')
run('src/rDCM_three_group_treatment_analysis.m')
run('src/rDCM_treatment_pre_post_analysis.m')

for svm modeling:
run('src/svm_routine.m')

the rest of .m files are helper functions
```
### 📝 NOTE  
- Preprocessing fMRI data can take a **very long time**. On the local machine used, it took approximately **8–12 hours** to preprocess everything
- appendix folder contains 2 codes that were not used for the final conclusions because they didn't show significant results (hypotheses_overall_results.m, figures in that folder obtained from here) or very slow accuracy predictions (appendix/svm_treat_pred.m). Nevertheless, they are part of the analysis development and/or future research.

## Authors 

Areli Balderas Maza: abalderas@student.ethz.ch 
David Blickenstorfer: davidbl@student.ethz.ch 
Zhining Zhang: zhinzhang@student.ethz.ch

## Acknowledgment

The authors thanks Dr. Heinzle Jakob for his supervising during the project, Computational Psychiatry Lab for providing the rDCM toolbox, and the developers of SPM12 for providing neuroimaging tools.

## License
The code was developed at University of Zurich and ETH Zurich and is part of the course: **Translational Neuromodeling, 227-0973-00L**. 