rng(1);

t=readtable('cmyc_results_with_names.txt','NumHeaderLines',0,'ReadVariableNames',false,'Delimiter','');
tt=readtable('bcl2_results_with_names.txt','NumHeaderLines',0,'ReadVariableNames',false,'Delimiter','');
for i=16:3:size(t,1) % skip Monte-Carlo
    cmyc_names=str2num(t.Var1{i});
    cmyc_gt=str2num(t.Var1{i+1})>=40;
    cmyc_pr=str2num(t.Var1{i+2})>=40;

    for j=7:3:size(tt,1) % skip Monte-Carlo
        bcl2_names=str2num(tt.Var1{j});
        bcl2_gt=str2num(tt.Var1{j+1})>=50;
        bcl2_pr=str2num(tt.Var1{j+2})>=50;

        [c,ia,ib]=intersect(cmyc_names,bcl2_names);
        gt=cmyc_gt(ia)&bcl2_gt(ib);
        pr=cmyc_pr(ia)&bcl2_pr(ib);
        
        sens=zeros(1000,1);
        spes=zeros(1000,1);
        for k=1:1000
            r=randsample(1:length(gt),length(gt),true);
            cm=confusionmat(gt(r),pr(r));
            spes(k)=cm(1,1)/sum(cm(1,:));
            sens(k)=cm(2,2)/sum(cm(2,:));
        end
        fprintf('%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
            mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
            mean(spes),prctile(spes,2.5),prctile(spes,97.5));
    end
end