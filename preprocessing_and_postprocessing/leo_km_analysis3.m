d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.efs_stat=t.efs_stat;
slides.efs_months=t.t_efs_mth;
slides.label=t.cmyc_wsi_score_dj;

gtz_cmyc=[];
prz_cmyc=[];
efz_stat_cmyc=[];
efz_months_cmyc=[];
idz_cmyc=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    efs_stat=[];
    efs_months=[];
    ids=[];
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
        ids=cat(1,ids,slides.slide_id(j));
    end
    gtz_cmyc=cat(1,gtz_cmyc,gts);
    prz_cmyc=cat(1,prz_cmyc,prs);
    efz_stat_cmyc=cat(1,efz_stat_cmyc,efs_stat);
    efz_months_cmyc=cat(1,efz_months_cmyc,efs_months);
    idz_cmyc=cat(1,idz_cmyc,ids);
end

d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
slides.efs_stat=t.efs_stat;
slides.efs_months=t.t_efs_mth;
slides.label=t.bcl2_wsi_score_dj;

gtz_bcl2=[];
prz_bcl2=[];
efz_stat_bcl2=[];
efz_months_bcl2=[];
idz_bcl2=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    efs_stat=[];
    efs_months=[];
    ids=[];
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
        ids=cat(1,ids,slides.slide_id(j));
    end
    gtz_bcl2=cat(1,gtz_bcl2,gts);
    prz_bcl2=cat(1,prz_bcl2,prs);
    efz_stat_bcl2=cat(1,efz_stat_bcl2,efs_stat);
    efz_months_bcl2=cat(1,efz_months_bcl2,efs_months);
    idz_bcl2=cat(1,idz_bcl2,ids);
end

[c,ia,ib]=intersect(idz_cmyc,idz_bcl2);


gtz_cmyc=gtz_cmyc(ia);
prz_cmyc=prz_cmyc(ia);
efz_stat_cmyc=efz_stat_cmyc(ia);
efz_months_cmyc=efz_months_cmyc(ia);

gtz_bcl2=gtz_bcl2(ib);
prz_bcl2=prz_bcl2(ib);
efz_stat_bcl2=efz_stat_bcl2(ib);
efz_months_bcl2=efz_months_bcl2(ib);

de=(prz_cmyc>=40)&(prz_bcl2>=70);
nde=~de;
logrank(cat(2,efz_months_cmyc(de),efz_stat_cmyc(de)==1),cat(2,efz_months_cmyc(nde),efz_stat_cmyc(nde)==1));


logrank(cat(2,efz_months_cmyc(prz_cmyc>=40),efz_stat_cmyc(prz_cmyc>=40)==1),cat(2,efz_months_cmyc(prz_cmyc<40),efz_stat_cmyc(prz_cmyc<40)==1));
logrank(cat(2,efz_months_bcl2(prz_bcl2>=70),efz_stat_bcl2(prz_bcl2>=70)==1),cat(2,efz_months_bcl2(prz_bcl2<70),efz_stat_bcl2(prz_bcl2<70)==1));
