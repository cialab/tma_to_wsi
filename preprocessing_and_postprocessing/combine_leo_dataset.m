t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
rd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/results/bcl2/feats/h5_files/';
wd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/results/bcl2_combined/feats/h5_files/';
slide_ids=[];
case_ids=[];
labels=[];
for i=1:size(t,1)
    slides=t.bcl2_files{i};
    if isempty(slides)
        continue;
    end

    slides=strsplit(slides,',');
    feats=[];
    for s=1:length(slides)
        ss=strsplit(slides{s},'.');
        f=h5read(strcat(rd,ss{1},'.h5'),'/features');
        feats=cat(2,feats,f);
    end

    label=t.bcl2_wsi_score_dj(i);
    h5create(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/features',size(feats),'Datatype','single');
    h5write(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/features',feats);
    h5create(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/label',size(label));
    h5write(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/label',label);
    slide_ids=cat(1,slide_ids,t.deid_id(i));
    case_ids=cat(1,case_ids,t.deid_id(i));
    labels=cat(1,labels,label);
end

t=table;
t.slide_id=slide_ids;
t.case_id=case_ids;
t.label=labels;
writetable(t,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/bcl2_score.csv');

t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
rd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/results/cmyc/feats/h5_files/';
wd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/results/cmyc_combined/feats/h5_files/';
slide_ids=[];
case_ids=[];
labels=[];
for i=1:size(t,1)
    slides=t.cmyc_files{i};
    if isempty(slides)
        continue;
    end

    slides=strsplit(slides,',');
    feats=[];
    for s=1:length(slides)
        ss=strsplit(slides{s},'.');
        f=h5read(strcat(rd,ss{1},'.h5'),'/features');
        feats=cat(2,feats,f);
    end

    label=t.bcl2_wsi_score_dj(i);
    h5create(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/features',size(feats),'Datatype','single');
    h5write(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/features',feats);
    h5create(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/label',size(label));
    h5write(strcat(wd,strcat(num2str(t.deid_id(i)),'.h5')),'/label',label);
    slide_ids=cat(1,slide_ids,t.deid_id(i));
    case_ids=cat(1,case_ids,t.deid_id(i));
    labels=cat(1,labels,label);
end

t=table;
t.slide_id=slide_ids;
t.case_id=case_ids;
t.label=labels;
writetable(t,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/cmyc_score.csv');
