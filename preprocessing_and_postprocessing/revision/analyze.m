rng(1);

exps=dir('results/*');
exps=exps(3:end);

for i=3:4
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

for i=1:2
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

i=3;
d=dir(fullfile(exps(i).folder,exps(i).name,'*.csv'));
splits=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/stanford_myc_by_patient_10foldcv/*.csv');
cmyc_gt=cell(length(d),1);
cmyc_pr=cell(length(d),1);
cmyc_names=cell(length(d),1);
for j=1:length(d)
    t=readtable(fullfile(d(j).folder,d(j).name));
    cmyc_gt{j}=t.teY;
    cmyc_pr{j}=t.teYp;

    t=readtable(fullfile(splits(j).folder,splits(j).name));
    cmyc_names{j}=t.test(~isnan(t.test));
end
cmyc_gt=cat(1,cmyc_gt{:})>=40;
cmyc_pr=cat(1,cmyc_pr{:})>=40;
cmyc_names=cat(1,cmyc_names{:});

i=1;
d=dir(fullfile(exps(i).folder,exps(i).name,'*.csv'));
splits=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/stanford_bcl2_by_patient_10foldcv/*.csv');
bcl2_gt=cell(length(d),1);
bcl2_pr=cell(length(d),1);
for j=1:length(d)
    t=readtable(fullfile(d(j).folder,d(j).name));
    bcl2_gt{j}=t.teY;
    bcl2_pr{j}=t.teYp;

    t=readtable(fullfile(splits(j).folder,splits(j).name));
    bcl2_names{j}=t.test(~isnan(t.test));
end
bcl2_gt=cat(1,bcl2_gt{:})>=50;
bcl2_pr=cat(1,bcl2_pr{:})>=50;
bcl2_names=cat(1,bcl2_names{:});

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
fprintf('\n%0.4f [%0.4f,%0.4f]\t%0.4f [%0.4f,%0.4f]\n',...
    mean(sens),prctile(sens,2.5),prctile(sens,97.5),...
    mean(spes),prctile(spes,2.5),prctile(spes,97.5));
