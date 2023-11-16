rng(1);
t=readtable('cmyc_results.txt','NumHeaderLines',0,'ReadVariableNames',false);
for i=1:3:size(t,1)
    gt=str2num(t.Var1{i+1});
    pr=str2num(t.Var1{i+2});

    corrs=zeros(1000,1);
    sens=zeros(1000,1);
    spes=zeros(1000,1);
    for j=1:1000
        r=randsample(1:length(gt),length(gt),true);
        corrs(j)=corr2(gt(r),pr(r));
        cm=confusionmat(gt(r)>=40,pr(r)>=40);
        spes(j)=cm(1,1)/sum(cm(1,:));
        sens(j)=cm(2,2)/sum(cm(2,:));
    end
    fprintf('%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
        mean(corrs),prctile(corrs,2.5),prctile(corrs,97.5),...
        mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
        mean(spes),prctile(spes,2.5),prctile(spes,97.5));
end

t=readtable('bcl2_results.txt','NumHeaderLines',0,'ReadVariableNames',false);
for i=1:3:size(t,1)
    gt=str2num(t.Var1{i+1});
    pr=str2num(t.Var1{i+2});

    corrs=zeros(1000,1);
    sens=zeros(1000,1);
    spes=zeros(1000,1);
    for j=1:1000
        r=randsample(1:length(gt),length(gt),true);
        corrs(j)=corr2(gt(r),pr(r));
        cm=confusionmat(gt(r)>=50,pr(r)>=50);
        spes(j)=cm(1,1)/sum(cm(1,:));
        sens(j)=cm(2,2)/sum(cm(2,:));
    end
    fprintf('%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
        mean(corrs),prctile(corrs,2.5),prctile(corrs,97.5),...
        mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
        mean(spes),prctile(spes,2.5),prctile(spes,97.5));
end