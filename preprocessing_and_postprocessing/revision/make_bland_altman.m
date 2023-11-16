exps=dir('results/*');
exps=exps(3:end);

% MYC 20x (u-MIL)
i=3;
d=dir(fullfile(exps(i).folder,exps(i).name,'*.csv'));
gt=cell(length(d),1);
pr=cell(length(d),1);
for j=1:length(d)
    t=readtable(fullfile(d(j).folder,d(j).name));
    gt{j}=t.teY;
    pr{j}=t.teYp;
end
gt=cat(1,gt{:});
pr=cat(1,pr{:});
[~,fh]=BlandAltman(gt,pr,{'pathologist','model','scores'},'CMYC, mean pooling');
saveas(fh,'CMYC-mean.png');

% BCL2 20x (u-MIL)
i=1;
d=dir(fullfile(exps(i).folder,exps(i).name,'*.csv'));
gt=cell(length(d),1);
pr=cell(length(d),1);
for j=1:length(d)
    t=readtable(fullfile(d(j).folder,d(j).name));
    gt{j}=t.teY;
    pr{j}=t.teYp;
end
gt=cat(1,gt{:});
pr=cat(1,pr{:});
[~,fh]=BlandAltman(gt,pr,{'pathologist','model','scores'},'BCL2, mean pooling');
saveas(fh,'BCL2-mean.png');

% CMYC 20x; att-MIL
t=readtable('../cmyc_results.txt','NumHeaderLines',0,'ReadVariableNames',false);
i=19;
gt=str2num(t.Var1{i+1})';
pr=str2num(t.Var1{i+2})';
[~,fh]=BlandAltman(gt,pr,{'pathologist','model','scores'},'CMYC, att pooling');
saveas(fh,'CMYC-att.png');

% BCL2 20x; att-MIL
t=readtable('../bcl2_results.txt','NumHeaderLines',0,'ReadVariableNames',false);
i=13;
gt=str2num(t.Var1{i+1})';
pr=str2num(t.Var1{i+2})';
[~,fh]=BlandAltman(gt,pr,{'pathologist','model','scores'},'BCL2, att pooling');
saveas(fh,'BCL2-att.png');

% CMYC; WSI
d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
gtz=[];
prz=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        gts=cat(1,gts,slides.label(j));
        pr=tt.Y_hat;
        pr(pr<0)=0;
        pr(pr>100)=100;
        prs=cat(1,prs,median(pr));

    end
    prz=cat(2,prz,prs);
    gtz=cat(2,gtz,gts);
end
gtz=mean(gtz,2);
prz=mean(prz,2);
[~,fh]=BlandAltman(gtz,prz,{'pathologist','model','scores'},'CMYC, WSI');
saveas(fh,'CMYC-WSI.png');

d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.bcl2_wsi_score_dj;
gtz=[];
prz=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        gts=cat(1,gts,slides.label(j));
        pr=tt.Y_hat;
        pr(pr<0)=0;
        pr(pr>100)=100;
        prs=cat(1,prs,median(pr));
    end
    prz=cat(2,prz,prs);
    gtz=cat(2,gtz,gts);
end
gtz=mean(gtz,2);
prz=mean(prz,2);
[~,fh]=BlandAltman(gtz,prz,{'pathologist','model','scores'},'BCL2, WSI');
saveas(fh,'BCL2-WSI.png');
