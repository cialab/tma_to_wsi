function attention_maps(stain)

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

slide_dir='/isilon/datalake/cialab/original/cialab/image_database/d00124/';

if strcmp(stain,'CMYC')

% CMYC
weights_dir='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas_A/';

d=dir('../leo_feats_tmas/cmyc/*.h5');
s=cellfun(@(x) strsplit(x,'_'),{d.name},'UniformOutput',false);
for i=1:length(s); s{i}=s{i}{1}; end
s=string(s);
u=unique(s,'stable');
for i=0:9
    for j=1:length(u)
        dd=dir(strcat('../leo_feats_tmas/cmyc/',u(j),'_*.h5'));
        k=1;
        if length(char(u(j)))==3; z='0'; elseif length(char(u(j)))==2; z='00'; end
        if length(strsplit(dd(k).name,'_'))==3; ss={'_01','_02'}; sss={'01','02'}; else; ss={''}; sss={'01'}; end

        for si=1:length(ss)
            if exist(strcat('wsi_heatmaps/cmyc/fold_',num2str(i),'/',u(j),ss{si},'.mat'),'file')
                fprintf('%s done; skip\n',u(j));
                continue;
            end
            dd=dir(strcat('../leo_feats_tmas/cmyc/',u(j),ss{si},'_*.h5'));

            As=cell(length(dd),1);
            xs=cell(length(dd),1);
            ys=cell(length(dd),1);
            for k=1:length(dd)
                xs{k}=h5read(fullfile(dd(k).folder,dd(k).name),'/xs');
                ys{k}=h5read(fullfile(dd(k).folder,dd(k).name),'/ys');
                As{k}=h5read(strcat(weights_dir,'fold_',num2str(i),'/',dd(k).name),'/A_raw');
            end
            xs=cat(1,xs{:});
            ys=cat(1,ys{:});
            As=cat(1,As{:});

            slide_path=char(strcat(slide_dir,z,u(j),'_CMYC_',sss{si},'.ndpi'));
            fp=openslide_open(char(strcat(slide_dir,z,u(j),'_CMYC_',sss{si},'.ndpi')));
            [w,h]=openslide_get_level0_dimensions(fp);

            hm=zeros(h/448,w/448,'single');
            xss=(xs+224)./448;
            yss=(ys+224)./448;
            A=As-min(As);
            A=A./max(A);
            A=1-A;
            for k=1:length(xss)
                hm(round(yss(k)),round(xss(k)))=A(k);
            end
            hm2=ordfilt2(hm,15,strel('disk',3,0).Neighborhood);
            save(strcat('wsi_heatmaps/cmyc/fold_',num2str(i),'/',u(j),ss{si},'.mat'),'xs','ys','As','A','hm','hm2','slide_path');
            fprintf('Done with %s\n',u(j));
        end
    end
end

elseif strcmp(stain,'BCL2')

% BCL2
weights_dir='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas_A/';

d=dir('../leo_feats_tmas/bcl2/20x/*.h5');
s=cellfun(@(x) strsplit(x,'_'),{d.name},'UniformOutput',false);
for i=1:length(s); s{i}=s{i}{1}; end
s=string(s);
u=unique(s,'stable');
for i=0:9
    for j=1:length(u)
        dd=dir(strcat('../leo_feats_tmas/bcl2/20x/',u(j),'_*.h5'));
        k=1;
        if length(char(u(j)))==3; z='0'; elseif length(char(u(j)))==2; z='00'; end
        if length(strsplit(dd(k).name,'_'))==3; ss={'_01','_02'}; sss={'01','02'}; else; ss={''}; sss={'01'}; end

        for si=1:length(ss)
            if exist(strcat('wsi_heatmaps/bcl2/fold_',num2str(i),'/',u(j),ss{si},'.mat'),'file')
                fprintf('%s done; skip\n',u(j));
                continue;
            end
            dd=dir(strcat('../leo_feats_tmas/bcl2/20x/',u(j),ss{si},'_*.h5'));

            As=cell(length(dd),1);
            xs=cell(length(dd),1);
            ys=cell(length(dd),1);
            for k=1:length(dd)
                xs{k}=h5read(fullfile(dd(k).folder,dd(k).name),'/xs');
                ys{k}=h5read(fullfile(dd(k).folder,dd(k).name),'/ys');
                As{k}=h5read(strcat(weights_dir,'fold_',num2str(i),'/',dd(k).name),'/A_raw');
            end
            xs=cat(1,xs{:});
            ys=cat(1,ys{:});
            As=cat(1,As{:});
    
            slide_path=char(strcat(slide_dir,z,u(j),'_BCL2_',sss{si},'.ndpi'));
            fp=openslide_open(char(strcat(slide_dir,z,u(j),'_BCL2_',sss{si},'.ndpi')));
            [w,h]=openslide_get_level0_dimensions(fp);
    
            hm=zeros(h/448,w/448,'single');
            xss=(xs+224)./448;
            yss=(ys+224)./448;
            A=As-min(As);
            A=A./max(A);
            A=1-A;
            for k=1:length(xss)
                hm(round(yss(k)),round(xss(k)))=A(k);
            end
            hm2=ordfilt2(hm,15,strel('disk',3,0).Neighborhood);
            save(strcat('wsi_heatmaps/bcl2/fold_',num2str(i),'/',u(j),ss{si},'.mat'),'xs','ys','As','A','hm','hm2','slide_path');
            fprintf('Done with %s\n',u(j));
        end
    end
end

end

end