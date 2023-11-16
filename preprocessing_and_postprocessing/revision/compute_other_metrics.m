rng(1);
types={'1-1','1-k','C-1','C-k','A-1','A-k'};

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

iccs=zeros(1000,6);
ps=zeros(1000,6);
for j=1:1000
    r=randsample(1:length(gt),length(gt),true);
    for i=1:length(types)
        [rr,~,~,~,~,~,p]=ICC(cat(2,gt(r),pr(r)),types{i});
        iccs(j,i)=rr;
        ps(j,i)=p;
    end
end
fprintf('CMYC\tu-MIL\t');
for i=1:length(types)
    fprintf('%0.4f [%0.4f,%0.4f]\t',mean(iccs(:,i)),prctile(iccs(:,i),2.5),prctile(iccs(:,i),97.5));
end
fprintf('\n')

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

iccs=zeros(1000,6);
ps=zeros(1000,6);
for j=1:1000
    r=randsample(1:length(gt),length(gt),true);
    for i=1:length(types)
        [rr,~,~,~,~,~,p]=ICC(cat(2,gt(r),pr(r)),types{i});
        iccs(j,i)=rr;
        ps(j,i)=p;
    end
end
fprintf('BCL2\tu-MIL\t');
for i=1:length(types)
    fprintf('%0.4f [%0.4f,%0.4f]\t',mean(iccs(:,i)),prctile(iccs(:,i),2.5),prctile(iccs(:,i),97.5));
end
fprintf('\n');

% CMYC 20x; att-MIL
t=readtable('../cmyc_results.txt','NumHeaderLines',0,'ReadVariableNames',false);
i=19;
gt=str2num(t.Var1{i+1})';
pr=str2num(t.Var1{i+2})';

iccs=zeros(1000,6);
ps=zeros(1000,6);
for j=1:1000
    r=randsample(1:length(gt),length(gt),true);
    for i=1:length(types)
        [rr,~,~,~,~,~,p]=ICC(cat(2,gt(r),pr(r)),types{i});
        iccs(j,i)=rr;
        ps(j,i)=p;
    end
end
fprintf('CMYC2\tatt-MIL\t');
for i=1:length(types)
    fprintf('%0.4f [%0.4f,%0.4f]\t',mean(iccs(:,i)),prctile(iccs(:,i),2.5),prctile(iccs(:,i),97.5));
end
fprintf('\n');

% BCL2 20x; att-MIL
t=readtable('../bcl2_results.txt','NumHeaderLines',0,'ReadVariableNames',false);
i=13;
gt=str2num(t.Var1{i+1})';
pr=str2num(t.Var1{i+2})';

iccs=zeros(1000,6);
ps=zeros(1000,6);
for j=1:1000
    r=randsample(1:length(gt),length(gt),true);
    for i=1:length(types)
        [rr,~,~,~,~,~,p]=ICC(cat(2,gt(r),pr(r)),types{i});
        iccs(j,i)=rr;
        ps(j,i)=p;
    end
end
fprintf('BCL2\tatt-MIL\t');
for i=1:length(types)
    fprintf('%0.4f [%0.4f,%0.4f]\t',mean(iccs(:,i)),prctile(iccs(:,i),2.5),prctile(iccs(:,i),97.5));
end
fprintf('\n');








% WSIs MYC
d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
all_gts=[];
all_prs=[];
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
    all_gts=cat(1,all_gts,gts);
    all_prs=cat(1,all_prs,prs);
end
iccs=zeros(1000,6);
ps=zeros(1000,6);
for j=1:1000
    r=randsample(1:length(all_gts),length(all_gts),true);
    for i=1:length(types)
        [rr,~,~,~,~,~,p]=ICC(cat(2,all_gts(r),all_prs(r)),types{i});
        iccs(j,i)=rr;
        ps(j,i)=p;
    end
end
fprintf('CMYC\tWSI\t');
for i=1:length(types)
    fprintf('%0.4f [%0.4f,%0.4f]\t',mean(iccs(:,i)),prctile(iccs(:,i),2.5),prctile(iccs(:,i),97.5));
end
fprintf('\n');

% WSI; BCL2
d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.bcl2_wsi_score_dj;
all_gts=[];
all_prs=[];
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
    all_gts=cat(1,all_gts,gts);
    all_prs=cat(1,all_prs,prs);
end
iccs=zeros(1000,6);
ps=zeros(1000,6);
for j=1:1000
    r=randsample(1:length(all_gts),length(all_gts),true);
    for i=1:length(types)
        [rr,~,~,~,~,~,p]=ICC(cat(2,all_gts(r),all_prs(r)),types{i});
        iccs(j,i)=rr;
        ps(j,i)=p;
    end
end
fprintf('BCL2\tWSI\t');
for i=1:length(types)
    fprintf('%0.4f [%0.4f,%0.4f]\t',mean(iccs(:,i)),prctile(iccs(:,i),2.5),prctile(iccs(:,i),97.5));
end
fprintf('\n');