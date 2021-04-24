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

    StylizedPath = fullfile(strcat(imageFolderLocation,'-Shuffle'));
    
    mkdir(StylizedPath);

    for i = 1 : nOfCategories
        for t_mode = { 'test'}
            cur_mode = t_mode{1};
            mkdir(fullfile(StylizedPath, cur_mode, categories{i}));
        end
    end 

    for t_mode = {'test'}
        cur_mode = t_mode{1};
        
        for k = 1:nOfCategories
            category = categories{k};
            myFiles = dir(fullfile(imageFolderLocation, cur_mode, category, '*.jpg'));
            
            parfor j=1:length(myFiles)
                fileName = myFiles(j).name;
                fileName = fileName(1,1:size(fileName,2)-4);
                
                rgbImageFilePath = fullfile(imageFolderLocation, cur_mode, category, myFiles(j).name);
                stylizedOutputFilePath = fullfile(StylizedPath, cur_mode, category, strcat(fileName,'.png'));
                
                I = imread(rgbImageFilePath);
                s = size(I);
                
                if(length(s) ~= 3)
                    I=I(:,:,[1 1 1]);
                    s = size(I);
                end
                
                stylized = randomOrderColors(I);
                imwrite(stylized,stylizedOutputFilePath);
                
            end
        end
        
    end
    
end
