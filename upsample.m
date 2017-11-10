horizontal_EPIs = dir(fullfile('Sparse_test_horizontal_EPIs\', '*.png'));

for i = 1:length(horizontal_EPIs)
    EPI = imread(strcat('Sparse_test_horizontal_EPIs/', horizontal_EPIs(i).name));
    
    g_x=fspecial('gaussian',[1 17], 1.5);
    EPI = imfilter(EPI, g_x);
    upsampled_img = imresize(EPI, [17, 1400]);
    imwrite(uint8(upsampled_img), strcat('upsampled_EPIs2/', horizontal_EPIs(i).name));
end