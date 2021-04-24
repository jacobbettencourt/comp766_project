function [SegList, reducedContour] = getContour(binaryImage)
    addpath(genpath('Lib'));

    binaryImage = binaryImage(:,:,1);
    SegList  = GetConSeg( binaryImage );
    
    num_segs = size(SegList,2);

    reducedContour = ones(size(binaryImage));
    for k=1:num_segs
        
        for j=1:length(SegList{1,k}(:,1))
            row = SegList{1,k}(j,1);
            col = SegList{1,k}(j,2);
            reducedContour(row,col) = 0;
        end
    end
    
end