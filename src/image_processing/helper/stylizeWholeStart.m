function stylized = stylizeWholeStart(colorContour,rgb)
    [rowMax,colMax,~]=size(colorContour);

    % step size along t
    dt=0.25;
    nOfIteration = 200;
    u1 = im2double(rgb);
    colorContour = im2double(colorContour);
    
    for t = 0:dt:nOfIteration
        
        u_xx = u1(:,[2:colMax colMax],:) - 2*u1 + u1(:,[1 1:colMax-1],:);       % finite difference approximation for u_xx
        u_yy = u1([2:rowMax rowMax],:,:) - 2*u1 + u1([1 1:rowMax-1],:,:);    % finite difference approximation for u_yy
        u1 = u1 + dt*(u_xx+u_yy);
        % overlayed the original edge rgb values on the difussed image
        u1(colorContour~=1)= colorContour(colorContour~=1);

    end
    
    for t = 0:dt:2
        
        u_xx = u1(:,[2:colMax colMax],:) - 2*u1 + u1(:,[1 1:colMax-1],:);       % finite difference approximation for u_xx
        u_yy = u1([2:rowMax rowMax],:,:) - 2*u1 + u1([1 1:rowMax-1],:,:);    % finite difference approximation for u_yy
        u1 = u1 + dt*(u_xx+u_yy);

    end

    stylized = u1;
end

%Train and test on colored contours works quite well
    %Test on flipped colored contours
    %
%Train and test on flipped colored contours will work ?
%Train and test on stylized
%Train and test on flipped stylized


