function [ output_args ] = DrawIdxNeg(  )
src='./competitionImageDataset/testset/semifrontal/';
imgdir = './competitionImageDataset/testset/semifrontal/'
imgBB0 = dir([src,'*.JSBB_Select']);

for i= 1: length(imgBB0)
   JSBB_SelectName = [imgdir,imgBB0(i).name];
   JSBB_UpdateBBName =  strrep(JSBB_SelectName,'.JSBB_Select','.JSBB_Update'); 
   imgName = strrep(JSBB_SelectName,'.JSBB_Select','.jpg'); 
   IDX = load(JSBB_SelectName);
   if IDX~=-1
       continue;
   end
   if exist(JSBB_UpdateBBName)
       continue;
   end
   figure;
   imshow(imread(imgName)); hold on; 
   b=3
   while(b==3)
   title('click left button twice to select the bounding box points')
   [x2,y2,b]=ginput(2); close all;
   imshow(imread(imgName)); hold on; 
   rectangle('Position',[x2(1),y2(1), x2(2)-x2(1),y2(2)-y2(1)],'EdgeColor','r');
   title('confirm click left (redo click right)')
   [x,y,b]=ginput(1) 
   end
   close all;
   annotation=[1.01,x2(1),y2(1),x2(2),y2(2)];
   display(annotation)
   dlmwrite(JSBB_UpdateBBName,annotation,'delimiter',' ');
end

end


