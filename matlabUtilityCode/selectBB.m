function [  ] = selectBB( )
% perform bounding box selection
% auother: shengtao xiao. xiao_shengtao@u.nus.edu
% date: 24/03/2017
Folder =  './competitionImageDataset/testset/semifrontal/'
imgs=dir([Folder,'*.jpg']);
% display(length(imgs));

for i=1:1:length(imgs)
   imgName =  [Folder,imgs(i).name];
   bbSelectName = strrep(imgName,'.jpg','.JSBB_Select');
   if exist(bbSelectName)
       continue;
   end
       
   BBName = strrep(imgName,'.jpg','.JSBB');
   img = imread(imgName);
   BBs = load(BBName);
   if size(BBs,1)==0
        dlmwrite(bbSelectName,-1);
        continue;
   end
   Centers = [(BBs(:,4)+BBs(:,2))/2 ,(BBs(:,5)+BBs(:,3))/2];
   Centers =[Centers;1,1]
   b=3;
   colors = ['b','g','r','w','k'];
   while(b==3)
       displayInfo(img,BBs)
       title('select center with left click(if all bounding box wrong, select the top left point)')
       [x2,y2,b]=ginput(1); close all;
       idx=searchClosest(Centers,[x2,y2]);
       %%%%%%%%%%%%%%%%%%%%%%%%%%
       if size(BBs) ~= 1
           displayInfo(img,BBs); hold on;
           plot(Centers(idx,1),Centers(idx,2),[colors(idx),'*'],'MarkerSize',50);
           title('right click to reselect (left will save it)');
           [x,y,b]=ginput(1); close all;
       end
       %%%%%%%%%%%%%%%%%%%
   end
   
   display(imgName);
   if idx<=size(BBs,1)
     dlmwrite(bbSelectName,idx)
   else
       -1
       dlmwrite(bbSelectName,-1)
   end
end

end

function [] = displayInfo(img, BBs)
   figure; 
   colors = ['b','g','r','w','k'];
   imshow(img,'InitialMagnification', 2000); hold on;
%    imshow(img); hold on;
%    truesize(img,[1000 1000])
   [w,h]=size(img)
   for h=1:size(BBs,1)
       if h>5
           break;
       end
       BB = BBs(h,2:5);
       rectangle('Position',[BB(1),BB(2),BB(3)-BB(1),BB(4)-BB(2)],'EdgeColor',colors(h),'LineWidth',5*min(w,h)/2)
       plot((BB(3)+BB(1))/2,(BB(4)+BB(2))/2,[colors(h),'o'], 'LineWidth',5*min(w,h)/2);
   end
    plot(1,1,['c','o']);
end

 function [idx] = searchClosest(Centers,Pxy)
    [value, idx_All] =sort(sum(abs(Centers - repmat(Pxy,size(Centers,1),1)),2),'ascend');
    idx=idx_All(1);
 end
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
% 
%  while(b==3)
%        title('click twice')
%    [x2,y2,b]=ginput(2); close all;
%    imshow(imread(imgName)); hold on; 
%    rectangle('Position',[x2(1),y2(1), x2(2)-x2(1),y2(2)-y2(1)],'EdgeColor','r');
%    title('confirm')
%    [x,y,b]=ginput(1) 
%    end
% %    close all;
%    annotation=[1.01,x2(1),y2(1),x2(2),y2(2)];
%    dlmwrite(bbName,annotation,'delimiter','\t');