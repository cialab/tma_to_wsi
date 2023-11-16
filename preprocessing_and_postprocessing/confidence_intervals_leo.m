rng(1);

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
corrs=zeros(1000,1);
sens=zeros(1000,1);
spes=zeros(1000,1);
for j=1:1000
    r=randsample(1:length(all_gts),length(all_gts),true);
    corrs(j)=corr2(all_gts(r),all_prs(r));
    cm=confusionmat(all_gts(r)>=40,all_prs(r)>=40);
    spes(j)=cm(1,1)/sum(cm(1,:));
    sens(j)=cm(2,2)/sum(cm(2,:));
end
fprintf('%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
        mean(corrs),prctile(corrs,2.5),prctile(corrs,97.5),...
        mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
        mean(spes),prctile(spes,2.5),prctile(spes,97.5));

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
    corrs=cat(1,corrs,corr(gts,prs,'type','Pearson'));
    all_gts=cat(1,all_gts,gts);
    all_prs=cat(1,all_prs,prs);
end
corrs=zeros(1000,1);
sens=zeros(1000,1);
spes=zeros(1000,1);
for j=1:1000
    r=randsample(1:length(all_gts),length(all_gts),true);
    corrs(j)=corr2(all_gts(r),all_prs(r));
    cm=confusionmat(all_gts(r)>=50,all_prs(r)>=50);
    spes(j)=cm(1,1)/sum(cm(1,:));
    sens(j)=cm(2,2)/sum(cm(2,:));
end
fprintf('%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
        mean(corrs),prctile(corrs,2.5),prctile(corrs,97.5),...
        mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
        mean(spes),prctile(spes,2.5),prctile(spes,97.5));

