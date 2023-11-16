t=readtable('cmyc_results_with_names.txt','NumHeaderLines',0,'ReadVariableNames',false,'Delimiter','');
tt=readtable('bcl2_results_with_names.txt','NumHeaderLines',0,'ReadVariableNames',false,'Delimiter','');
pt=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00147/stanford_donors.csv');

i=16;
cmyc_names=str2num(t.Var1{i});
cmyc_gt=str2num(t.Var1{i+1})>=40;
cmyc_pr=str2num(t.Var1{i+2})>=40;

j=7;
bcl2_names=str2num(tt.Var1{j});
bcl2_gt=str2num(tt.Var1{j+1})>=50;
bcl2_pr=str2num(tt.Var1{j+2})>=50;

[c,ia,ib]=intersect(cmyc_names,bcl2_names);
gt=cmyc_gt(ia)&bcl2_gt(ib);
pr=cmyc_pr(ia)&bcl2_pr(ib);

[cc,iia,iib]=intersect(c,pt.donor);
gt=gt(iia); pr=pr(iia);
pfs=pt.pfs(iib); pfs_event=pt.pfs_event(iib);
os=pt.os(iib); os_event=pt.os_event(iib);

pfs_event2=cell(length(pfs_event),1);
pfs_event2(pfs_event==1)={'Progression'};
pfs_event2(pfs_event==0)={'NoProgression'};
group_var=cell(length(gt),1);
group_var(gt)={'double-expresser'};
group_var(~gt)={'not double-expresser'};
[p,fh,stats]=MatSurv(pfs,pfs_event2,group_var,'RT_YLabel',false,'legend',false);
saveas(fh,strcat('surv_plots/tma_pfs_de_path.png'));

pfs_event2=cell(length(pfs_event),1);
pfs_event2(pfs_event==1)={'Progression'};
pfs_event2(pfs_event==0)={'NoProgression'};
group_var=cell(length(pr),1);
group_var(pr)={'double-expresser'};
group_var(~pr)={'not double-expresser'};
[p,fh,stats]=MatSurv(pfs,pfs_event2,group_var,'RT_YLabel',false,'legend',false);
saveas(fh,strcat('surv_plots/tma_pfs_de_model.png'));

os_event2=cell(length(os_event),1);
os_event2(os_event==1)={'Dead'};
os_event2(os_event==0)={'Alive'};
group_var=cell(length(gt),1);
group_var(gt)={'double-expresser'};
group_var(~gt)={'not double-expresser'};
[p,fh,stats]=MatSurv(os,os_event2,group_var,'RT_YLabel',false,'legend',false);
saveas(fh,strcat('surv_plots/tma_os_de_path.png'));

os_event2=cell(length(os_event),1);
os_event2(os_event==1)={'Dead'};
os_event2(os_event==0)={'Alive'};
group_var=cell(length(pr),1);
group_var(pr)={'double-expresser'};
group_var(~pr)={'not double-expresser'};
[p,fh,stats]=MatSurv(os,os_event2,group_var,'RT_YLabel',false,'legend',false);
saveas(fh,strcat('surv_plots/tma_os_de_model.png'));


