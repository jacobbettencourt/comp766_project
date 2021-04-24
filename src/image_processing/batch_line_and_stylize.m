function [] = batch_contour_and_stylized(imageFolderLocation)
    addpath("helper");
    files = dir(fullfile(imageFolderLocation,'train'));
    % Extract only those that are directories.
    subFolders = files([files.isdir]);
    % Ignore . and ..
    categories = cell(length(subFolders)-2,1);
    
    for k = 3 : length(subFolders)
        categories{k-2} = subFolders(k).name;
    end
    
    categories = string(categories);
    nOfCategories = length(categories)

    LineDrawingPath = fullfile(strcat(imageFolderLocation,'-LineDrawing'));
    ColoredContourPath = fullfile(strcat(imageFolderLocation,'-ColoredContour'));
    StylizedPath = fullfile(strcat(imageFolderLocation,'-Stylized'));
    psPath = fullfile(strcat(imageFolderLocation,'-ps'));
    
    mkdir(LineDrawingPath);
    mkdir(ColoredContourPath);
    mkdir(StylizedPath);
    mkdir(psPath);

    for i = 1 : nOfCategories
        for t_mode = { 'val','train','test'}
            cur_mode = t_mode{1};
            mkdir(fullfile(LineDrawingPath, cur_mode, categories{i}));
            mkdir(fullfile(psPath,cur_mode,categories{i}));
            
            mkdir(fullfile(StylizedPath, cur_mode, categories{i}));
            mkdir(fullfile(ColoredContourPath,cur_mode,categories{i}));
        end
    end 

    for t_mode = {'test','train','val'}
        cur_mode = t_mode{1};
        
        for k = 1:nOfCategories
            category = categories{k};
            myFiles = dir(fullfile(imageFolderLocation, cur_mode, category, '*.jpg'));
            
            parfor j=1:length(myFiles)
                fileName = myFiles(j).name;
                fileName = fileName(1,1:size(fileName,2)-4);
                
                rgbImageFilePath = fullfile(imageFolderLocation, cur_mode, category, myFiles(j).name);
                psImageFilePath = fullfile(psPath,cur_mode,category,strcat(fileName,'.ps'));
                edgeImageFilePath = fullfile(LineDrawingPath,cur_mode,category,strcat(fileName,'.png'));
                contourOutputFilePath = fullfile(ColoredContourPath, cur_mode, category, strcat(fileName,'.png'));
                stylizedOutputFilePath = fullfile(StylizedPath, cur_mode, category, strcat(fileName,'.png'));
                
                I = imread(rgbImageFilePath);
                s = size(I);
                
                if(length(s) ~= 3)
                    I=I(:,:,[1 1 1]);
                    s = size(I);
                end
                
                %%% Logical Linear edges
                str = strcat('wine pgmloglin.exe -E ',{' '},rgbImageFilePath,{' '},psImageFilePath);
                command = str{1};

                [~,~]= system(command);

                ps2png(psImageFilePath,edgeImageFilePath,[s(1),s(2)]);

                edge = imread(edgeImageFilePath);

                %%%Thin edges
                edge = ~imbinarize(im2double(edge));
                edge = bwmorph(edge,'thin',Inf);
                edge = double(~bwmorph(edge,'thin',Inf));
                edge = cat(3,edge,edge,edge);

                coloredContour = contourColor(I,edge);
                imwrite(coloredContour,contourOutputFilePath);

                stylized = stylizeWholeStart(coloredContour,I);
                imwrite(stylized,stylizedOutputFilePath);
                
            end
            
            "Folder: " + cur_mode + " at category " + k + " (" +k/nOfCategories + ")"
            %Clean wasted spaced
            delete(fullfile(psPath,cur_mode,category,'*.*'));
        end
        
    end
    
end
