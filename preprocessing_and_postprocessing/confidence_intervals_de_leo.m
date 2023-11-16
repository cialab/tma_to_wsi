rng(1);

d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
all_gts=[];
all_prs=[];
all_names=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    names=[];
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
        names=cat(1,names,slides.slide_id(j));
    end
    all_gts=cat(1,all_gts,gts);
    all_prs=cat(1,all_prs,prs);
    all_names=cat(1,all_names,names);
end
cmyc_gt=all_gts>=40;
cmyc_pr=all_prs>=40;
cmyc_names=all_names;

d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.bcl2_wsi_score_dj;
all_gts=[];
all_prs=[];
all_names=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    names=[];
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
        names=cat(1,names,slides.slide_id(j));
    end
    all_gts=cat(1,all_gts,gts);
    all_prs=cat(1,all_prs,prs);
    all_names=cat(1,all_names,names);
end
bcl2_gt=all_gts>=50;
bcl2_pr=all_prs>=50;
bcl2_names=all_names;

% Hmm
cmyc_names=cmyc_names(1:52); cmyc_gt=cmyc_gt(1:52);
bcl2_names=bcl2_names(1:57); bcl2_gt=bcl2_gt(1:57);
cmyc_pr=sum(reshape(cmyc_pr,[52 10]),2);
bcl2_pr=sum(reshape(bcl2_pr,[57 10]),2);

[c,ia,ib]=intersect(cmyc_names,bcl2_names);
cmyc_gt=cmyc_gt(ia); cmyc_pr=cmyc_pr(ia);
bcl2_gt=bcl2_gt(ib); bcl2_pr=bcl2_pr(ib);
all_gts=cmyc_gt&bcl2_gt;
all_prs=(cmyc_pr>=5)&(bcl2_pr>=5);

sens=zeros(1000,1);
spes=zeros(1000,1);
j=1;
while j<=1000
    r=randsample(1:length(all_gts),length(all_gts),true);
    if sum(all_gts(r))==0
        continue;
    end
    cm=confusionmat(all_gts(r),all_prs(r));
    spes(j)=cm(1,1)/sum(cm(1,:));
    sens(j)=cm(2,2)/sum(cm(2,:));
    j=j+1;
end
fprintf('%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
        mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
        mean(spes),prctile(spes,2.5),prctile(spes,97.5));

