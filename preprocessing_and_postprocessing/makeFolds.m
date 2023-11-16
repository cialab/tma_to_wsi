t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/bcl2_efs.csv');
[cases,u]=unique(t.case_id);
labels=t.label(u);
labels=string(labels)=="yes";

rng(1);
pos_idx=find(labels==1);
neg_idx=find(labels==0);
idxs=[];
k=0;
while ~(isempty(pos_idx) && ~isempty(neg_idx))
    ipos=randi([1 length(pos_idx)]);
    ineg=randi([1 length(neg_idx)]);
    idxs=[idxs; pos_idx(ipos); neg_idx(ineg)];
    pos_idx(ipos)=[];
    neg_idx(ineg)=[];
    k=k+2;
end
idxs=[idxs; pos_idx; neg_idx];
cases=cases(idxs);
labels=labels(idxs);

s=0;
for i=1:2:k
    te={};
    ipos=find(cases(i)==t.case_id);
    for j=1:length(ipos)
        te=cat(1,te,t.slide_id(ipos(j)));
    end
    ineg=find(cases(i+1)==t.case_id);
    for j=1:length(ineg)
        te=cat(1,te,t.slide_id(ineg(j)));
    end
    tr={};
    for j=1:i-1
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    for j=i+2:length(cases)
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    nums=strsplit(num2str(0:length(tr)-1),' ');

    tt=table;
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=string(tr);
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('splits/bcl2_efs/splits_',num2str(s),'.csv'));
    s=s+1;
end

t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/bcl2_stage.csv');
[cases,u]=unique(t.case_id);
labels=t.label(u);
labels=string(labels)=="yes";

rng(1);
pos_idx=find(labels==1);
neg_idx=find(labels==0);
idxs=[];
k=0;
while ~(isempty(pos_idx) && ~isempty(neg_idx))
    ipos=randi([1 length(pos_idx)]);
    ineg=randi([1 length(neg_idx)]);
    idxs=[idxs; pos_idx(ipos); neg_idx(ineg)];
    pos_idx(ipos)=[];
    neg_idx(ineg)=[];
    k=k+2;
end
idxs=[idxs; pos_idx; neg_idx];
cases=cases(idxs);labels=labels(idxs);

s=0;
for i=1:2:k
    te={};
    ipos=find(cases(i)==t.case_id);
    for j=1:length(ipos)
        te=cat(1,te,t.slide_id(ipos(j)));
    end
    ineg=find(cases(i+1)==t.case_id);
    for j=1:length(ineg)
        te=cat(1,te,t.slide_id(ineg(j)));
    end
    tr={};
    for j=1:i-1
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    for j=i+2:length(cases)
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    nums=strsplit(num2str(0:length(tr)-1),' ');

    tt=table;
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=string(tr);
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('splits/bcl2_stage/splits_',num2str(s),'.csv'));
    s=s+1;
end

t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/cmyc_efs.csv');
[cases,u]=unique(t.case_id);
labels=t.label(u);
labels=string(labels)=="yes";

rng(1);
pos_idx=find(labels==1);
neg_idx=find(labels==0);
idxs=[];
k=0;
while ~(isempty(pos_idx) && ~isempty(neg_idx))
    ipos=randi([1 length(pos_idx)]);
    ineg=randi([1 length(neg_idx)]);
    idxs=[idxs; pos_idx(ipos); neg_idx(ineg)];
    pos_idx(ipos)=[];
    neg_idx(ineg)=[];
    k=k+2;
end
idxs=[idxs; pos_idx; neg_idx];
cases=cases(idxs);
labels=labels(idxs);

s=0;
for i=1:2:k
    te={};
    ipos=find(cases(i)==t.case_id);
    for j=1:length(ipos)
        te=cat(1,te,t.slide_id(ipos(j)));
    end
    ineg=find(cases(i+1)==t.case_id);
    for j=1:length(ineg)
        te=cat(1,te,t.slide_id(ineg(j)));
    end
    tr={};
    for j=1:i-1
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    for j=i+2:length(cases)
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    nums=strsplit(num2str(0:length(tr)-1),' ');

    tt=table;
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=string(tr);
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('splits/cmyc_efs/splits_',num2str(s),'.csv'));
    s=s+1;
end

t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/cmyc_stage.csv');
[cases,u]=unique(t.case_id);
labels=t.label(u);
labels=string(labels)=="yes";

rng(1);
pos_idx=find(labels==1);
neg_idx=find(labels==0);
idxs=[];
k=0;
while ~(isempty(pos_idx) && ~isempty(neg_idx))
    ipos=randi([1 length(pos_idx)]);
    ineg=randi([1 length(neg_idx)]);
    idxs=[idxs; pos_idx(ipos); neg_idx(ineg)];
    pos_idx(ipos)=[];
    neg_idx(ineg)=[];
    k=k+2;
end
idxs=[idxs; pos_idx; neg_idx];
cases=cases(idxs);
labels=labels(idxs);

s=0;
for i=1:2:k
    te={};
    ipos=find(cases(i)==t.case_id);
    for j=1:length(ipos)
        te=cat(1,te,t.slide_id(ipos(j)));
    end
    ineg=find(cases(i+1)==t.case_id);
    for j=1:length(ineg)
        te=cat(1,te,t.slide_id(ineg(j)));
    end
    tr={};
    for j=1:i-1
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    for j=i+2:length(cases)
        idxs=find(cases(j)==t.case_id);
        for idx=1:length(idxs)
            tr=cat(1,tr,t.slide_id(idxs(idx)));
        end
    end
    nums=strsplit(num2str(0:length(tr)-1),' ');

    tt=table;
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=string(tr);
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('splits/cmyc_stage/splits_',num2str(s),'.csv'));
    s=s+1;
end
