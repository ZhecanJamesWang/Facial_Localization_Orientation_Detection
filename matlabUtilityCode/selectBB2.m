function [  ] = selectBB( Folder )
% perform bounding box selection
% auother: shengtao xiao. xiao_shengtao@u.nus.edu
% date: 24/03/2017
Folder=  './profile/'
Folder=  './JamesPart/tmp/'
imgs=dir([Folder,'*.jpg']);

for i=1:1:length(imgs)
   imgName =  [Folder,imgs(i).name];
   bbSelectName = strrep(imgName,'.jpg','.JSBB_Select')
   BBName = strrep(imgName,'.jpg','.JSBB');
   BBMName = strrep(imgName,'.jpg','.JSBBM');
%%%%%%%  comment this to test -1 issue      
%    if exist(bbSelectName)
%        continue;
%    end
%%%%%%%  uncomment this to test -1 issue%%   
   IDX = load(bbSelectName)             %%
   if IDX~=-1                           %%
       continue;                        %%
   end                                  %%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       

% %%%%%%%  uncomment this to test -1 issue      
%    if exist(BBMName)                    %%
%        copyfile(BBMName,BBName)         %%
%    else                                 %%
%        continue;                        %%
%    end                                  %% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   img = imread(imgName);
   BBs = load(BBName);
   if size(BBs,1)==0
        dlmwrite(bbSelectName,-1);
        continue;
   end
   if size(BBs,1)>=5
   BBs=BBs(1:4,:);
   end
   Centers = [(BBs(:,4)+BBs(:,2))/2 ,(BBs(:,5)+BBs(:,3))/2];
   Centers =[Centers;1,1];
   b=3;
   colors = ['b','g','r','w','k','c','m','y'];
   while(b==3)
       displayInfo(img,BBs)
       title('select center with left click(if all bounding box wrong, select the top left point)')
       xlabel(sprintf('%d/%d',i,length(imgs)))
%        ylabel(sprintf('%d',IDX))
       [x2,y2,b]=ginput(1); close all;
       idx=searchClosest(Centers,[x2,y2]);
       displayInfo(img,BBs); hold on;
       plot(Centers(idx,1),Centers(idx,2),[colors(idx),'*'],'MarkerSize',12);
       title(sprintf('right click to reselect (left will save it) %d',idx));
       [x,y,b]=ginput(1); close all;
   end
   if idx<=size(BBs,1)
       idx
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
   imshow(img); hold on;
   [w,h]=size(img);
   for h=1:size(BBs,1)
       if h>5
           break;
       end
       BB = BBs(h,2:5);
       rectangle('Position',[BB(1),BB(2),BB(3)-BB(1),BB(4)-BB(2)],'EdgeColor',colors(h),'LineWidth',min(w,h)/2)
       plot((BB(3)+BB(1))/2,(BB(4)+BB(2))/2,[colors(h),'o']);
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