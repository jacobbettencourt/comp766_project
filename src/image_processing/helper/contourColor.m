function [colorContour] = contourColor(original, contour)
    colorContour = uint8(255*contour);
    
    for row=1:size(original,1)
        for col=1:size(original,2)
            if((contour(row,col) == 0) || (contour(row,col) == 1) && (oneInNeighborhood(contour,row,col) == 1))
                
                colorContour(row,col,1:3) = original(row,col,1:3);
            end
        end
    end
    
end

function [inNeigh] = oneInNeighborhood(contour,x,y)
    s = size(contour);
    maxX = s(1);
    maxY = s(2);
    inNeigh = 1;
    black = 0;
    
    if(x + 1 < maxX && contour(x+1,y) == black)
        return;
    elseif(x + 1 < maxX && y+1 < maxY && contour(x+1,y+1) == black)
        return;
    elseif(x + 1 < maxX && y - 1 > 0 && contour(x+1,y-1) == black)
        return;
    elseif(y + 1 < maxY && contour(x,y+1) == black)
        return;
    elseif(y-1 > 0 && contour(x,y-1) == black)
        return;
    elseif(x-1 > 0)
        if(contour(x-1,y) == black)
            return
        elseif(y+1 < maxY && contour(x-1,y+1) == black)
            return;
        elseif(y-1 > 0 && contour(x-1,y-1) == black)
            return;
        end
    end
    
    inNeigh=0;
    return;
end