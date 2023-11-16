im=imread('C:\Users\ttavolar\Documents\lymphoma\fig1\0101_BCL2_01.tif');
mask=bwareafilt(imfill(im(:,:,1)<200,'holes'),[200 Inf]);
[x,y]=find(mask);
k=kmeans(cat(2,x,y),500);
imshow(mask);
hold on;
for i=1:500
yy=y(k==i);
xx=x(k==i);
idx=boundary(yy,xx);
plot(yy(idx),xx(idx),'k');
end
hold off

img=zeros(size(mask),'logical');
for i=1:500
yy=y(k==i);
xx=x(k==i);
idx=boundary(yy,xx,1);
for j=1:length(idx)
img(xx(idx(j)),yy(idx(j)))=1;
end
end

imwrite(255.*uint8(img),'cluster_overlay.png','Alpha',double(img));