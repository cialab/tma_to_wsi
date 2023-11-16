d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.efs_stat=t.efs_stat;
slides.efs_months=t.t_efs_mth;
slides.label=t.cmyc_wsi_score_dj;

gtz=[];
prz=[];
efz_stat=[];
efz_months=[];
all_names=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    efs_stat=[];
    efs_months=[];
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
        efs_stat=cat(1,efs_stat,slides.efs_stat(j));
        efs_months=cat(1,efs_months,slides.efs_months(j));
        names=cat(1,names,slides.slide_id(j));
    end
    gtz=cat(1,gtz,gts);
    prz=cat(1,prz,prs);
    efz_stat=cat(1,efz_stat,efs_stat);
    efz_months=cat(1,efz_months,efs_months);
    all_names=cat(1,all_names,names);
end
cmyc_gt=gtz>=40;
cmyc_pr=prz>=40;
cmyc_names=all_names;
cmyc_time=efz_months;
cmyc_status=efz_stat;

d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
slides.efs_stat=t.efs_stat;
slides.efs_months=t.t_efs_mth;
slides.label=t.bcl2_wsi_score_dj;
gtz=[];
prz=[];
efz_stat=[];
efz_months=[];
all_names=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    efs_stat=[];
    efs_months=[];
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
        efs_stat=cat(1,efs_stat,slides.efs_stat(j));
        efs_months=cat(1,efs_months,slides.efs_months(j));
        names=cat(1,names,slides.slide_id(j));
    end
    gtz=cat(1,gtz,gts);
    prz=cat(1,prz,prs);
    efz_stat=cat(1,efz_stat,efs_stat);
    efz_months=cat(1,efz_months,efs_months);
    all_names=cat(1,all_names,names);
end
bcl2_gt=gtz>=50;
bcl2_pr=prz>=50;
bcl2_names=all_names;
bcl2_time=efz_months;
bcl2_status=efz_stat;

cmyc_names=cmyc_names(1:52); cmyc_gt=cmyc_gt(1:52);
bcl2_names=bcl2_names(1:57); bcl2_gt=bcl2_gt(1:57);
cmyc_pr=sum(reshape(cmyc_pr,[52 10]),2);
bcl2_pr=sum(reshape(bcl2_pr,[57 10]),2);

[c,ia,ib]=intersect(cmyc_names,bcl2_names);
gt=cmyc_gt(ia)&bcl2_gt(ib);
pr=(cmyc_pr(ia)>=5)&(bcl2_pr(ib)>=5);
cmyc_time=cmyc_time(ia); bcl2_time=bcl2_time(ib);
cmyc_status=cmyc_status(ia); bcl2_status=bcl2_status(ib);
efs=cmyc_time; % it's the same a bcl2
efs_event=cmyc_status; % it's the same as bcl2

efs_event2=cell(length(efs_event),1);
efs_event2(efs_event==2)={'Event'};
efs_event2(efs_event==1)={'NoEvent'};
group_var=cell(length(gt),1);
group_var(gt)={'double-expresser'};
group_var(~gt)={'not double-expresser'};
[p,fh,stats]=MatSurv(efs,efs_event2,group_var,'RT_YLabel',false,'legend',false);
saveas(fh,strcat('surv_plots/wsi_efs_de_path.png'));

efs_event2=cell(length(efs_event),1);
efs_event2(efs_event==2)={'Event'};
efs_event2(efs_event==1)={'NoEvent'};
group_var=cell(length(pr),1);
group_var(pr)={'double-expresser'};
group_var(~pr)={'not double-expresser'};
[p,fh,stats]=MatSurv(efs,efs_event2,group_var,'RT_YLabel',false,'legend',false);
saveas(fh,strcat('surv_plots/wsi_efs_de_model.png'));