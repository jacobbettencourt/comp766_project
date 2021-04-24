function [shuffled] = randomOrderColors(I)
%RANDOMORDERCOLORS Randomly permutes entire pixels
%   Uses randomperm to get the new locations in a 2d grid, but moves the
%   entire pixel
    I = im2double(I);
    r = I(:,:,1);
    g = I(:,:,2);
    b = I(:,:,3);
    
    shuffled = I;
    s = size(I);
    s = s(1:2);
    numbers = 1:s(1)*s(2);
    numbers = numbers(randperm(length(numbers)));
    
    for i=1:s(1)*s(2)
        col = floor(i/s(1))+1;
        row = i-s(1)*(col-1)+1;
        shuffled(row,col,1) = r(numbers(i));
        shuffled(row,col,2) = g(numbers(i));
        shuffled(row,col,3) = b(numbers(i));
    end
    
    subplot(1,2,1);
    imshow(shuffled);
    subplot(1,2,2);
    imshow(I);
end

