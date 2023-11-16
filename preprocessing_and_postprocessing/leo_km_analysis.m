t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');

% KM curve for CMYC clinical
figure;
hold on;
ecdf(t.t_efs_mth(t.cmyc_wsi_score_dj>=40),'Function','survivor','Censoring',t.efs_stat(t.cmyc_wsi_score_dj>=40)==1);
ecdf(t.t_efs_mth(t.cmyc_wsi_score_dj<40),'Function','survivor','Censoring',t.efs_stat(t.cmyc_wsi_score_dj<40)==1);
title('cmyc clinical')
hold off

ps=[];
ths=[];
for i=5:5:95
    [f1,x1]=ecdf(t.t_efs_mth(t.cmyc_wsi_score_dj>=i),'Function','survivor','Censoring',t.efs_stat(t.cmyc_wsi_score_dj>=i)==1);
    [f2,x2]=ecdf(t.t_efs_mth(t.cmyc_wsi_score_dj<i),'Function','survivor','Censoring',t.efs_stat(t.cmyc_wsi_score_dj<i)==1);
    [h1,p1,ks2stat] = kstest2(f1,f2,'Tail','larger');
    ths=[ths i];
    ps=[ps p1];
end

% KM curve for BCL2 clinical
figure;
hold on;
ecdf(t.t_efs_mth(t.bcl2_wsi_score_dj>=70),'Function','survivor','Censoring',t.efs_stat(t.bcl2_wsi_score_dj>=70)==1);
ecdf(t.t_efs_mth(t.bcl2_wsi_score_dj<70),'Function','survivor','Censoring',t.efs_stat(t.bcl2_wsi_score_dj<70)==1);
title('bcl2 clinical')
hold off

ps=[];
ths=[];
for i=25:5:95
    [f1,x1]=ecdf(t.t_efs_mth(t.bcl2_wsi_score_dj>=i),'Function','survivor','Censoring',t.efs_stat(t.bcl2_wsi_score_dj>=i)==1);
    [f2,x2]=ecdf(t.t_efs_mth(t.bcl2_wsi_score_dj<i),'Function','survivor','Censoring',t.efs_stat(t.bcl2_wsi_score_dj<i)==1);
    [h1,p1,ks2stat] = kstest2(f1,f2,'Tail','larger');
    ths=[ths i];
    ps=[ps p1];
end

% KM curve for double expressor
figure;
hold on;
dp=(t.bcl2_wsi_score_dj>=50) & (t.cmyc_wsi_score_dj>=40);
n=~dp;
ecdf(t.t_efs_mth(dp),'Function','survivor','Censoring',t.efs_stat(dp)==1);
ecdf(t.t_efs_mth(n),'Function','survivor','Censoring',t.efs_stat(n)==1);
title('double expressor clinical')
hold off;