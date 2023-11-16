d=dir('stanford_features/bcl2_40_40x/*.h5');
case_id=[];
slide_id=[];
label=[];
for j=1:length(d)
    s=strsplit(d(j).name,'.'); s=string(s{1});
    case_id=cat(1,case_id,s);
    slide_id=cat(1,slide_id,s);
    l=h5read(fullfile(d(j).folder,d(j).name),'/label');
    label=cat(1,label,string(num2str(l)));
end
t=table;
t.case_id=case_id;
t.slide_id=slide_id;
t.label=label;
writetable(t,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/stanford_bcl2.csv');

wd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/stanford_bcl2/';
s1=round(size(t,1)*0.1);
s2=2*s1;
for f=1:100
    rng(f);
    tt=t(randperm(size(t,1)),:);
    tr=tt(1:end-s2,:);
    vl=tt(end-s2+1:end-s1,:);
    te=tt(end-s1+1:end,:);
    

    ttt=table;
    ttt.(' ')=[0:size(tr,1)-1]';
    ttt.train=tr.slide_id;
    ttt.val=cat(1,vl.slide_id,repmat([""],[size(tr,1)-size(vl,1) 1]));
    ttt.test=cat(1,te.slide_id,repmat([""],[size(tr,1)-size(te,1) 1]));
    writetable(ttt,strcat(wd,'splits_',num2str(f-1),'.csv'));
end

d=dir('stanford_features/bcl2_40_40x_by_patient/*.h5');
case_id=[];
slide_id=[];
label=[];
for j=1:length(d)
    s=strsplit(d(j).name,'.'); s=string(s{1});
    case_id=cat(1,case_id,s);
    slide_id=cat(1,slide_id,s);
    l=h5read(fullfile(d(j).folder,d(j).name),'/label');
    label=cat(1,label,string(num2str(l)));
end
t=table;
t.case_id=case_id;
t.slide_id=slide_id;
t.label=label;
writetable(t,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/stanford_bcl2_by_patient.csv');

wd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/stanford_bcl2_by_patient/';
s1=round(size(t,1)*0.1);
s2=2*s1;
for f=1:100
    rng(f);
    tt=t(randperm(size(t,1)),:);
    tr=tt(1:end-s2,:);
    vl=tt(end-s2+1:end-s1,:);
    te=tt(end-s1+1:end,:);
    

    ttt=table;
    ttt.(' ')=[0:size(tr,1)-1]';
    ttt.train=tr.slide_id;
    ttt.val=cat(1,vl.slide_id,repmat([""],[size(tr,1)-size(vl,1) 1]));
    ttt.test=cat(1,te.slide_id,repmat([""],[size(tr,1)-size(te,1) 1]));
    writetable(ttt,strcat(wd,'splits_',num2str(f-1),'.csv'));
end

% LOO CV
wd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/stanford_bcl2_by_patient_loo/';
s=round(size(t,1)*0.15);
for f=1:size(t,1)
    rng(f);
    tt=t;
    te=tt(f,:);
    tt(f,:)=[];

    idxs=[];
    for i=0:10:100
        if i==40 || i==60
            continue;
        end
        idx=find(tt.label==num2str(i));
        idxs=[idxs idx(randi([1 length(idx)]))];
    end
    tr=tt(idxs,:);
    tt(idxs,:)=[];

    idxs=[];
    addit=0;
    for i=0:10:100
        if i==40 || i==60
            continue;
        end
        idx=find(tt.label==num2str(i));
        if isempty(idx)
            addit=1;
            continue
        end
        idxs=[idxs idx(randi([1 length(idx)]))];
    end
    vl=tt(idxs,:);
    if addit
        vl=cat(1,vl,tr(5,:));
    end
    tt(idxs,:)=[];

    tt=tt(randperm(size(tt,1)),:);
    tr=cat(1,tr,tt(1:end-(s-10),:));
    vl=cat(1,vl,tt(end-(s-10)+1:end,:));

    ttt=table;
    ttt.(' ')=[0:size(tr,1)-1]';
    ttt.train=tr.slide_id;
    ttt.val=cat(1,vl.slide_id,repmat([""],[size(tr,1)-size(vl,1) 1]));
    ttt.test=cat(1,te.slide_id,repmat([""],[size(tr,1)-size(te,1) 1]));
    writetable(ttt,strcat(wd,'splits_',num2str(f-1),'.csv'));
end

% 10-fold for Lee Cooper
idxs=zeros(size(t,1),1);
vals=zeros(size(t,1),1);
rng(1);
rs=0:10:100;
rs=rs(randperm(11));
for r=1:11
    i=rs(r);
    idx=find(t.label==num2str(i));
    while ~isempty(idx)
        l=randsample(length(idx),1);
        j=idx(l);
        idx(l)=[];
        l=randsample(find(idxs==0),1);
        if sum(vals==i)==0 % first entry
            idxs(l)=j;
            vals(l)=i;
        else
            max_d=0;
            max_idx=0;
            for k=1:length(idxs)
                if idxs(k)==0
                    d=sum(abs(k-find(vals==i)));
                    if max_d<d
                        max_d=d;
                        max_idx=k;
                    end
                end
            end
            idxs(max_idx)=j;
            vals(max_idx)=i;
        end
    end
end
tt=t(idxs,:);
wd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/stanford_bcl2_by_patient_10foldcv/';
l=round(linspace(0,size(tt,1),11));
for f=1:10
    tr=cat(1,tt(1:l(f),:),tt(l(f+1)+1:end,:));
    vl=tr;
    te=tt(l(f)+1:l(f+1),:);
    
    ttt=table;
    ttt.(' ')=[0:size(tr,1)-1]';
    ttt.train=tr.slide_id;
    ttt.val=cat(1,vl.slide_id,repmat([""],[size(tr,1)-size(vl,1) 1]));
    ttt.test=cat(1,te.slide_id,repmat([""],[size(tr,1)-size(te,1) 1]));
    writetable(ttt,strcat(wd,'splits_',num2str(f-1),'.csv'));
end