function ps2png(fileName,outputFileName,imsize)
threshold_contour_size = 10;
str_threshold = 0;
fid = fopen(fileName,'r');
nOfLines = linecount(fileName);

tline = fgetl(fid);
lineNum = 1;
nOfPoints = nOfLines - 92;
X = zeros(nOfPoints,1);
Y = zeros(nOfPoints,1);
DIR = zeros(nOfPoints,8);
Edge = zeros(nOfPoints,1);
counter = 1;
visited_point_matrix = zeros(imsize);
edge_matrix = zeros(imsize);
xx = [-1 0 1 -1 1 -1 0 1];
yy = [-1 -1 -1 0 0 1 1 1];

while lineNum < (nOfLines-2)
    tline = fgetl(fid);
    if lineNum >= 90
        
        newStr = extractBetween(tline,1,length(tline)-5);
        SS = sscanf(newStr{1},'%d');

        y0 = SS(1);
        x0 = SS(2);
        dir = SS(3);
        dir = dir+1;
        if dir>8
            dir = dir-8;
        end
        
        strength = SS(4);
        if(strength > str_threshold)
        x = min(max(x0,1),imsize(1)) ;
        y = min(max(y0,1),imsize(2)) ;
        if(x~=1 && y~=1 && x~=imsize(1)-1 && y~=imsize(2)-1)
        if(visited_point_matrix(x,y)==0)
            X(counter) = x;
            Y(counter) = y;
            DIR(counter,dir) = 1;   
            Edge(counter) = strength;
            visited_point_matrix(x,y) = counter;
            edge_matrix(x,y) = max(edge_matrix(x,y),strength);
            counter = counter+1;
        else
            index_counter = visited_point_matrix(x,y);
            DIR(index_counter,dir) = 1;
            
        end
        end
        end
    end
    lineNum = lineNum + 1; 
end
M = edge_matrix;
M = bwareaopen(M,threshold_contour_size);
M = edge_matrix.*M;
fI = (1./M).^0.5;
imwrite(fI,strcat(outputFileName));
end

